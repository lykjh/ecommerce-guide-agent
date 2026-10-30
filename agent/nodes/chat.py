"""闲聊节点 + 推荐节点。

闲聊是整张图里**唯一累积购买意向信号**的链路：
每轮闲聊并行做两件事——生成回复、更新信号强度；信号攒到 SIGNAL_THRESHOLD 后，
由 main_agent 的条件边转到 recommend_node 顺势推荐。

为什么不把信号判断做成 START 的分流：
信号是「跨轮累积的状态」，不是「这一轮该干什么」。业务咨询、商品对比这些
明确意图根本不需要判信号，做成 START 分支等于每轮都多调一次模型。
"""

import re

from langchain_core.messages import AIMessage

from agent.deps import client, retriever
from agent.memory import AgentState, memory_manager
from agent.prompts import CHAT_PROBE_SYSTEM, CHAT_SYSTEM, RECOMMEND_FALLBACK
from agent.router import SIGNAL_NONE, SIGNAL_PROBE_THRESHOLD, SIGNAL_THRESHOLD, signal_router
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.chat")


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
    response = await client.ainvoke(msgs)

    #二道保险：提示词管不住的模型，在这里把编造的清单砍掉（详见函数注释）
    stripped = _strip_fabricated_block(response.content)
    if not stripped.strip():
        stripped = "您想看点什么呢？告诉我大概的品类和预算，我帮您挑。"
    response.content = stripped

    logger.info(
        f"✅ chat_node 回答完成 | {len(response.content)} 字符 | "
        f"信号 {name}/{level} | 档位 {'探需' if prompt is CHAT_PROBE_SYSTEM else '闲聊'}"
    )
    return {"messages": [response], "signal_state": signal_state}


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
    response = await client.ainvoke(msgs)
    logger.info(f"✅ recommend_node 推荐完成 | {len(response.content)} 字符")

    #🔴 推荐完必须归零：信号还留在阈值以上的话，用户下一句随便聊一句就又被推一次
    return {
        "messages": [response],
        "signal_state": {"signal": SIGNAL_NONE, "level": 0},
    }
