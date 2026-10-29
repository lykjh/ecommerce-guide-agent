"""闲聊节点 + 推荐节点。

闲聊是整张图里**唯一累积购买意向信号**的链路：
每轮闲聊并行做两件事——生成回复、更新信号强度；信号攒到 SIGNAL_THRESHOLD 后，
由 main_agent 的条件边转到 recommend_node 顺势推荐。

为什么不把信号判断做成 START 的分流：
信号是「跨轮累积的状态」，不是「这一轮该干什么」。业务咨询、商品对比这些
明确意图根本不需要判信号，做成 START 分支等于每轮都多调一次模型。
"""

from langchain_core.messages import AIMessage

from agent.deps import client, retriever
from agent.memory import AgentState, memory_manager
from agent.prompts import CHAT_PROBE_SYSTEM, CHAT_SYSTEM, RECOMMEND_FALLBACK
from agent.router import SIGNAL_NONE, SIGNAL_PROBE_THRESHOLD, SIGNAL_THRESHOLD, signal_router
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.chat")


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
