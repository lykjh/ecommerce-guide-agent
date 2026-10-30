"""闲聊节点 + 推荐节点。

闲聊是整张图里**唯一累积购买意向信号**的链路：
每轮闲聊先判信号强度、再按档位生成回复；信号攒到 SIGNAL_THRESHOLD 后，
由 main_agent 的条件边转到 recommend_node 顺势推荐。

为什么不把信号判断做成 START 的分流：
信号是「跨轮累积的状态」，不是「这一轮该干什么」。业务咨询、商品对比这些
明确意图根本不需要判信号，做成 START 分支等于每轮都多调一次模型。

🔴 本文件里所有 LLM 调用都挂了 INTERNAL_TAG
==============================================
闲聊/推荐这两档的输出都要先过防幻觉守卫（复核 -> 必要时重写）才能给用户。
如果不打 tag，langgraph 会把「生成过程」的 token 直接流到界面上，
编造内容就先跑出去了，事后重写根本来不及。打上 tag 后：
生成过程不可见 -> 守卫定稿 -> 作为 AIMessage 返回 ->
由 main_agent 的兜底路径切片吐出（切片是为了保住打字机效果）。
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.deps import INTERNAL_TAG, client, extract_client, retriever
from agent.memory import AgentState, memory_manager
from agent.prompts import (
    CHAT_GUARD_SYSTEM,
    CHAT_PROBE_SYSTEM,
    CHAT_REPAIR_SYSTEM,
    CHAT_SAFE_FALLBACK,
    CHAT_SYSTEM,
    RECOMMEND_EXTRACT_SYSTEM,
    RECOMMEND_FALLBACK,
    RECOMMEND_REPAIR_SYSTEM,
)
from agent.router import SIGNAL_NONE, SIGNAL_PROBE_THRESHOLD, SIGNAL_THRESHOLD, signal_router
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.chat")

#生成与复核都走「内部调用」：不打 tag 会被 stream_mode="messages" 当成回答流给用户
chat_client = client.with_config({"tags": [INTERNAL_TAG]})
#复核用便宜的快模型（就是判信号那个），只让它答「有/无」，延迟几乎无感
guard_client = extract_client.with_config({"tags": [INTERNAL_TAG]})


#清单行的行首标记：• · ▪ - * 1. 1) ① 之类
_LIST_LINE_RE = re.compile(r"\s*(?:[•·▪▫◦]|[-*+]|[0-9]+[.、)）]|[①-⑩])\s*\S")
#清单前面那句"引子"，典型说法："顺便悄悄说：我们库里刚好有几款…"
_LEAD_IN_WORDS = ("库里", "仓库", "刚好", "几款", "推荐", "悄悄说", "顺便", "看看哪款", "这些商品", "热销")


def _strip_fabricated_block(text: str) -> str:
    """砍掉闲聊回复里凭空冒出来的商品清单。

    🔴 为什么需要这道保险：chat_node **不检索任何商品**，回复里一旦出现列表，
    几乎可以断定是模型自己脑补的（实测"今天好热啊"被编出三款降温神器，
    还配了"正在被抢购🔥"）。真要推荐必须走 recommend_node——那条链路才带检索上下文。

    做法：从第一个清单行开始整段砍掉，连它上面那句引子一起砍（引子往往才是
    编造的起点："我们库里刚好有几款…"）。只砍清单块，前面的正常寒暄保留。
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not _LIST_LINE_RE.match(line):
            continue

        head = lines[:i]
        #砍掉紧邻的引子行（往上找最近一个非空行，命中引导词就一并丢掉）
        for j in range(len(head) - 1, -1, -1):
            if head[j].strip():
                if any(w in head[j] for w in _LEAD_IN_WORDS):
                    head = head[:j]
                break

        kept = "\n".join(head).rstrip()
        logger.warning(
            f"🚫 闲聊回复里出现疑似编造的商品清单，已截断 | "
            f"{len(text)} 字符 -> {len(kept)} 字符 | 砍掉: {line.strip()[:30]}"
        )
        return kept
    return text


async def _mentions_product(text: str) -> bool:
    """复核：这段回复里有没有私自提到具体商品。

    闲聊档的规则很简单粗暴——**任何商品都不许出现**，所以不用去商品库比对，
    只要判断「有没有提商品」就行。这也是为什么敢用便宜模型：任务很轻。
    """
    resp = await guard_client.ainvoke([
        SystemMessage(content=CHAT_GUARD_SYSTEM),
        HumanMessage(content=text),
    ])
    answer = (resp.content or "").strip()
    if not answer.startswith(("有", "无")):
        #模型答非所问就按「无」处理，不能因为守卫自己抽风就把正常回复干掉
        logger.warning(f"⚠️ 守卫模型返回了意外结果「{answer[:20]}」，按「无」处理")
        return False
    return answer.startswith("有")


async def _rewrite_without_product(msgs) -> str:
    """上一版编了商品，重写一版只寒暄提问的（保留历史与用户输入）。"""
    resp = await chat_client.ainvoke([
        SystemMessage(content=CHAT_REPAIR_SYSTEM),
        *msgs[1:],
    ])
    return (resp.content or "").strip()


async def chat_node(state: AgentState):
    #⚠️ 信号判断必须排在回复生成**之前**：本轮用「纯闲聊」还是「探需」话术，
    #取决于本轮信号强度。并行就只能拿上轮信号来决定本轮语气，慢一拍——
    #用户第一句就流露意愿时，客服会毫无反应。这里用一次串行等待换正确的档位。
    signal_state = await signal_router.judge(state)
    level = signal_state.get("level", 0)
    name = signal_state.get("signal", SIGNAL_NONE)
    has_intent = name not in (SIGNAL_NONE, "明确拒绝")

    #三档话术：没意愿纯闲聊 / 有意愿主动探需 / 意愿够强交给 recommend_node 正式推荐
    if has_intent and SIGNAL_PROBE_THRESHOLD <= level < SIGNAL_THRESHOLD:
        prompt = CHAT_PROBE_SYSTEM
        logger.info(f"🔍 探需档 | 信号 {name}/{level}，回复改为主动询问预算与用途")
    else:
        #level 已达推荐阈值时用普通闲聊简短回应，正式推荐交给后面的 recommend_node，
        #避免这里问一遍需求、推荐节点又推一遍
        prompt = CHAT_SYSTEM

    msgs = await memory_manager.build_chat_messages(state, prompt)
    response = await chat_client.ainvoke(msgs)

    #防幻觉两道闸：
    #1) 零成本的规则预检——列表型编造（• 冰感凉枕…）直接砍掉，省掉一次模型调用
    #2) 模型复核——行内列举型（"我们家有脆脆薯片、香浓坚果"）规则抓不到，交给守卫
    content = _strip_fabricated_block(response.content)

    if content.strip() and await _mentions_product(content):
        logger.warning(f"🚫 闲聊回复疑似编造商品，触发重写 | 原文: {content[:60]}")
        content = await _rewrite_without_product(msgs)
        if await _mentions_product(content):
            logger.warning(f"🚫 重写后仍在提商品，改用安全兜底 | 重写版: {content[:60]}")
            content = CHAT_SAFE_FALLBACK

    if not content.strip():
        content = CHAT_SAFE_FALLBACK
    response.content = content

    logger.info(
        f"✅ chat_node 回答完成 | {len(response.content)} 字符 | "
        f"信号 {name}/{level} | 档位 {'探需' if prompt is CHAT_PROBE_SYSTEM else '闲聊'}"
    )
    return {"messages": [response], "signal_state": signal_state}


async def _product_names(text: str) -> list[str]:
    """从推荐回复里抽出它提到的商品名，用于和 context 做证据比对。"""
    resp = await guard_client.ainvoke([
        SystemMessage(content=RECOMMEND_EXTRACT_SYSTEM),
        HumanMessage(content=text),
    ])
    raw = (resp.content or "").strip()
    if not raw or raw.startswith("无"):
        return []
    return [n.strip() for n in re.split(r"[,，、;；\n]", raw) if n.strip()]


def _evidence_missing(names: list[str], context: str) -> list[str]:
    """evidence 校验：回复里提到的商品名，必须在检索到的 context 里找得到。

    比让模型自检更硬——这是能举证的：查不到的名字就是凭空冒出来的。

    ⚠️ 不能直接整串比对：抽取出来的常是「品牌 + 品名」（初汀 Toproto 主动降噪蓝牙耳机），
    而 context 里品牌和品名是分开写的，整串匹配会**误判真商品为编造**（实测踩到过）。
    所以按片段拆开，只要任一片段（≥3 字）能在 context 里找到就算有证据。
    """
    missing = []
    for name in names:
        if name in context:
            continue
        #整串找不到就退一步：品牌+品名连写（"初汀主动降噪耳机"）或模型简写商品名时，
        #context 里往往只有其中的一段。用 4 字滑动窗口，只要有一段能对上就算有出处；
        #真编造的名字（"月光限定桂花乌龙茶"）一段都对不上。
        size = 4
        if len(name) <= size or not any(
            name[i:i + size] in context for i in range(len(name) - size + 1)
        ):
            missing.append(name)
    return missing


async def _rewrite_recommend(msgs, context: str) -> str:
    """上一版推荐里混进了库里没有的商品，按 context 重写。"""
    resp = await chat_client.ainvoke([
        SystemMessage(content=RECOMMEND_REPAIR_SYSTEM.format(context=context)),
        *msgs[1:],
    ])
    return (resp.content or "").strip()


async def recommend_node(state: AgentState):
    user_query = state["messages"][-1].content

    #推荐必须基于真实商品，否则就是让模型凭空编——比答不上来更糟
    context = retriever.get_context(user_query)

    if not context:
        logger.warning("⚠️ 推荐时没检索到商品上下文，改为反问用户具体需求，不硬编商品")
        return {
            "messages": [AIMessage(content=RECOMMEND_FALLBACK)],
            "signal_state": {"signal": SIGNAL_NONE, "level": 0},
        }

    msgs = await memory_manager.build_recommend_messages(state, user_query, context)
    response = await chat_client.ainvoke(msgs)

    #evidence 校验：回复里点名的商品，必须能在检索到的商品信息里找到
    content = response.content
    missing = _evidence_missing(await _product_names(content), context)
    if missing:
        logger.warning(f"🚫 推荐里出现库里没有的商品 {missing}，触发重写")
        content = await _rewrite_recommend(msgs, context)
        missing = _evidence_missing(await _product_names(content), context)
        if missing:
            logger.warning(f"🚫 重写后仍有编造 {missing}，改用兜底话术")
            content = RECOMMEND_FALLBACK

    response.content = content
    logger.info(f"✅ recommend_node 推荐完成 | {len(response.content)} 字符")

    #🔴 推荐完必须归零：信号还留在阈值以上的话，用户下一句随便聊一句就又被推一次
    return {
        "messages": [response],
        "signal_state": {"signal": SIGNAL_NONE, "level": 0},
    }
