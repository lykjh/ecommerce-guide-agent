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

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.deps import DB_PATH, INTERNAL_TAG
from agent.memory import AgentState, memory_manager
from agent.nodes.business import business_node
from agent.nodes.chat import chat_node, recommend_node
from agent.nodes.compare import compare_shop_node
from agent.nodes.total import total_node
from agent.prompts import OUTPUT_FALLBACK
from agent.router import SIGNAL_NONE, SIGNAL_THRESHOLD, router
from utils.log_tool import get_loger, set_session_id

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
    "商品推荐": "recommend_node",
}

#会产出「给用户看的内容」的节点，流式输出只放行这些
VISIBLE_NODES = ("business_node", "chat_node", "total_node", "compare_shop_node", "recommend_node")


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


def should_recommend(state) -> str:
    """闲聊之后判断是否该顺势推荐。

    信号强度攒到阈值就转推荐节点，否则照常收尾。
    ⚠️ 「明确拒绝」要单独挡掉：它的 level 被强制为 0，本来就不会过阈值，
    这里再判一次是为了防止以后改动强度规则时误触。
    """
    signal = state.get("signal_state") or {}
    level = signal.get("level", 0)
    name = signal.get("signal", SIGNAL_NONE)

    if level >= SIGNAL_THRESHOLD and name not in (SIGNAL_NONE, "明确拒绝"):
        logger.info(f"🎯 信号达标 {name}/{level}（阈值 {SIGNAL_THRESHOLD}）-> 触发推荐")
        return "recommend_node"

    logger.info(f"➡️ 信号未达标 {name}/{level}（阈值 {SIGNAL_THRESHOLD}）-> 直接收尾")
    return "summarize_node"


# ============================================================
# 二、装配图
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("business_node", business_node)
builder.add_node("chat_node", chat_node)
builder.add_node("total_node", total_node)
builder.add_node("compare_shop_node", compare_shop_node)
builder.add_node("recommend_node", recommend_node)
#摘要节点由 memory_manager 提供，所有「真实对话」节点都要汇到它，否则这轮不进记忆
builder.add_node("summarize_node", memory_manager.summarize_node)

#⚠️ router 不能注册成普通节点：
#节点返回值必须是 dict（用来更新 state），而 router 返回的是 "闲聊"/"业务咨询" 这类字符串，
#注册成节点会直接报 InvalidUpdateError: Expected dict, got 闲聊。
#正确姿势是让它只当 START 的「分支函数」——只负责挑下一个节点，不产生 state 更新。
#这样每轮也只调用一次路由模型（注册成节点会被调用两次：进节点一次、判分支一次）。
builder.add_conditional_edges(START, route_with_fallback, ROUTE_MAP)

builder.add_edge("business_node", "summarize_node")
builder.add_edge("total_node", "summarize_node")
builder.add_edge("compare_shop_node", "summarize_node")
#闲聊走条件边：信号攒够了先推荐再收尾，没攒够直接收尾
builder.add_conditional_edges("chat_node", should_recommend)
#推荐节点自己会把信号归零（见 nodes/chat.py），避免用户每说一句就被推一次
builder.add_edge("recommend_node", "summarize_node")

builder.add_edge("summarize_node", END)


# ============================================================
# 三、流式输出
# ============================================================

async def get_ai_stream(user_input: str, thread_id: str = "default"):
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("请输入非空文本")
    if len(user_input) > memory_manager.MAX_HISTORY_CHARS:
        raise ValueError("输入过长，请缩短到 6000 字符以内")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("会话 ID 不能为空")
    #把会话 id 绑到日志上下文：之后 router / 节点 / retriever 打的所有日志都会带上它，
    #排查某轮对话时直接 grep 这个 id 就能拉出完整链路，不用再拿时间戳去对齐几个文件。
    set_session_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"▶️ 新一轮对话 | 输入「{user_input[:40]}」")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        emitted = False
        #本轮节点交出的定稿内容（chat/recommend 走防幻觉守卫，生成过程不可见，
        #只能等节点结束后从它的返回值里拿）
        final_reply = ""

        #🔴 为什么同时订阅 messages 和 updates 两种模式：
        #messages 用来把「边生成边输出」的 token 实时流出去（打字机效果）；
        #updates 用来拿到节点**本轮返回的 state 更新**，里面带着 final_reply。
        #以前只用 messages，拿不到定稿就只能回头去 state 的 messages 里倒着找最后一条
        #AIMessage——那是在猜，多轮会话下会把上一轮的回复再吐一遍。
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode=["messages", "updates"]
        ):

            if mode == "messages":
                msg_chunk, metadata = chunk

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

            #updates 模式下 chunk 形如 {"节点名": 该节点返回的 state 更新}
            elif mode == "updates":
                for node_name, update in (chunk or {}).items():
                    if node_name not in VISIBLE_NODES or not isinstance(update, dict):
                        continue
                    if update.get("final_reply"):
                        final_reply = update["final_reply"]

        #3) 没有流式输出（节点返回的是定稿消息）时，用节点交出的定稿切片吐出
        if not emitted:
            if final_reply:
                logger.info(f"ℹ️ 本轮无流式输出，改用节点定稿 | {len(final_reply)} 字符")
                #切片吐出：整段一次性砸出来太生硬，切小块保住打字机效果
                for i in range(0, len(final_reply), 12):
                    yield final_reply[i:i + 12]
            else:
                #🔴 绝不静默。以前这里什么都不 yield，界面就是一片空白，
                #日志里连个 ERROR 都没有，完全没法定位。
                logger.error(
                    "❌ 本轮没有任何可输出内容：既没有流式 chunk，"
                    "节点也没交出 final_reply。已用兜底话术顶上"
                )
                yield OUTPUT_FALLBACK


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
