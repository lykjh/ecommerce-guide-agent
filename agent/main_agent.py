"""图装配 + 流式输出。

这个文件只负责两件事：
1. 把节点拼成一张图（谁接谁、意图怎么分流）
2. 把节点的输出流给前端

**任何业务逻辑都不放这里**——节点在 agent/nodes/，提示词在 agent/prompts.py，
取数在 tool/。新增意图 = 写个 nodes/xxx.py + 在 ROUTE_MAP 加一行，本文件不用动。
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.deps import DB_PATH, INTERNAL_TAG
from agent.memory import AgentState, memory_manager
from agent.nodes.business import business_node
from agent.nodes.chat import chat_node
from agent.nodes.compare import compare_shop_node
from agent.nodes.total import total_node
from agent.router import router
from utils.log_tool import get_loger

logger = get_loger("agent.main")


# ============================================================
# 一、意图 -> 节点 的映射
# ============================================================

#新增意图要同时改三处：这里、router 的 RouteName、prompts.py 的 ROUTE_SYSTEM 候选列表
ROUTE_MAP = {
    "闲聊": "chat_node",
    "业务咨询": "business_node",
    "概览筛选": "total_node",
    "商品对比": "compare_shop_node",
}

#会产出「给用户看的内容」的节点，流式输出只放行这些
VISIBLE_NODES = ("business_node", "chat_node", "total_node", "compare_shop_node")


async def route_with_fallback(state) -> str:
    """在 router 外面包一层兜底。

    🔴 不加这层会直接崩：router 的 prompt 里列了 7 类意图，但只要返回的意图不在
    ROUTE_MAP 里，LangGraph 会抛 `KeyError: '商品推荐'`，整轮对话直接挂掉。
    兜底策略是记日志 + 转业务咨询（RAG 至少能答），而不是默默吞掉——
    日志里会留下「哪个意图没接」，方便后面补实现。
    """
    intent = await router.router(state)
    if intent not in ROUTE_MAP:
        logger.warning(
            f"⚠️ 意图「{intent}」尚未接入实现，兜底到业务咨询。已实现: {sorted(ROUTE_MAP)}"
        )
        return "业务咨询"
    logger.info(f"🧭 路由结果: {intent} -> {ROUTE_MAP[intent]}")
    return intent


# ============================================================
# 二、装配图
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("business_node", business_node)
builder.add_node("chat_node", chat_node)
builder.add_node("total_node", total_node)
builder.add_node("compare_shop_node", compare_shop_node)
#摘要节点由 memory_manager 提供，所有「真实对话」节点都要汇到它，否则这轮不进记忆
builder.add_node("summarize_node", memory_manager.summarize_node)

#⚠️ router 不能注册成普通节点：
#节点返回值必须是 dict（用来更新 state），而 router 返回的是 "闲聊"/"业务咨询" 这类字符串，
#注册成节点会直接报 InvalidUpdateError: Expected dict, got 闲聊。
#正确姿势是让它只当 START 的「分支函数」——只负责挑下一个节点，不产生 state 更新。
#这样每轮也只调用一次路由模型（注册成节点会被调用两次：进节点一次、判分支一次）。
builder.add_conditional_edges(START, route_with_fallback, ROUTE_MAP)

builder.add_edge("business_node", "summarize_node")
builder.add_edge("chat_node", "summarize_node")
builder.add_edge("total_node", "summarize_node")
builder.add_edge("compare_shop_node", "summarize_node")
builder.add_edge("summarize_node", END)


# ============================================================
# 三、流式输出
# ============================================================

async def get_ai_stream(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"▶️ 新一轮对话 | thread={thread_id} | 输入「{user_input[:40]}」")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        emitted = False

        async for msg_chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):

            #1) 先挡掉「内部调用」：参数抽取这类中间过程不能让用户看到。
            #   stream_mode="messages" 捕获的是节点内每一次 LLM 调用，不是每个节点一次，
            #   所以只按节点名过滤是不够的，必须按调用粒度挡。
            if INTERNAL_TAG in (metadata.get("tags") or []):
                continue

            #2) 再按节点过滤：路由模型自己输出的意图标签归属 __start__，也要挡掉
            node_name = metadata.get("langgraph_node")
            if (
                msg_chunk.type == "AIMessageChunk"
                and msg_chunk.content
                and node_name in VISIBLE_NODES
            ):
                emitted = True
                yield msg_chunk.content

        #3) 兜底：节点直接 return AIMessage(...) 的「写死回复」（例如"没找到该商品"）
        #   不经过 LLM，stream_mode="messages" 不会产出任何 chunk，用户会看到一片空白。
        #   这种情况从最终状态里把最后一条 AI 回复取出来一次性吐出去。
        if not emitted:
            logger.warning("⚠️ 本轮无流式输出（节点返回的是预设回复），改为从状态里取最终消息")
            try:
                final_state = await graph.aget_state(config)
                for msg in reversed(final_state.values.get("messages", [])):
                    #排除 AIMessageChunk：那是流式的碎片，不是完整回复
                    if isinstance(msg, AIMessage) and not isinstance(msg, AIMessageChunk) and msg.content:
                        yield msg.content
                        break
            except Exception as e:
                logger.exception(f"❌ 取最终消息失败: {e}")


if __name__ == "__main__":
    async def test():
        print("测试开始：")
        async for chunk in get_ai_stream("无线蓝牙耳机多少钱"):
            print(chunk, end="", flush=True)
        print("\n")

        # 测一下记忆：换个问法问上面那句话
        async for chunk in get_ai_stream("智能保温杯还有货吗"):
            print(chunk, end="", flush=True)
        print("\n")

    asyncio.run(test())
