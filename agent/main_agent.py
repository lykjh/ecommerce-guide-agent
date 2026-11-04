"""图装配与输出：意图节点写入分类，条件边按 router.ROUTE_MAP 分流。"""

import asyncio
from contextlib import aclosing, suppress
import threading
from uuid import uuid4
from weakref import WeakValueDictionary

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.deps import DB_PATH, INTERNAL_TAG
from agent.memory import memory_manager
from agent.state import AgentState
from agent.telemetry import TurnUsage
from agent.budget import TurnBudget, BudgetStop, BUDGET_REPLY, budget_scope
from agent.nodes.business import business_node
from agent.nodes.chat import chat_node
from agent.nodes.recommend import recommend_node
from agent.nodes.compare import compare_shop_node
from agent.nodes.total import total_node
from agent.nodes.routing import route_reply_node
from agent.prompts import OUTPUT_FALLBACK
from agent.router import ROUTE_MAP, SIGNAL_NONE, SIGNAL_THRESHOLD, router, route_with_fallback
from agent.utils.log_tool import get_loger, set_session_id

logger = get_loger("agent.main")
get_loger("agent.telemetry")
get_loger("agent.memory")
get_loger("agent.router")
get_loger("agent.budget")

# shortcut: 仅覆盖同一进程内的线程和事件循环，多实例部署前需共享接纳控制。
_session_guard = threading.Lock()
_active_sessions: set[str] = set()
SESSION_BUSY_REPLY = "上一条消息还在处理中，请等回复完成后再发送这条消息。"
# 同一运行循环、同一数据库共用IO锁；最后一条连接退出后不保留锁或循环引用。
_checkpoint_locks: WeakValueDictionary = WeakValueDictionary()


# ============================================================
# 一、意图 -> 节点 的映射
# ============================================================

BUSINESS_NODES = (business_node, chat_node, total_node, compare_shop_node, recommend_node, route_reply_node)
VISIBLE_NODES = tuple(node.__name__ for node in BUSINESS_NODES)


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

builder.add_node("intent_node", router.node)
for node in BUSINESS_NODES:
    builder.add_node(node.__name__, node)
# 业务节点汇入摘要；固定边界回复仍由 checkpointer 保存消息，但不额外调用模型。
builder.add_node("summarize_node", memory_manager.summarize_node)

# 路由节点写入本轮 intent；条件边只读取结果，不再重复调用模型。
builder.add_edge(START, "intent_node")
builder.add_conditional_edges("intent_node", route_with_fallback, ROUTE_MAP)
# 固定边界回应不调用摘要模型；历史上下文仍由记忆窗口限制。
builder.add_edge("route_reply_node", END)

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
    async with aclosing(stream_reply(user_input, thread_id)) as stream:
        async for chunk in stream:
            yield chunk


async def stream_reply(user_input: str, thread_id: str, request_id: str | None = None):
    """消费入口已校验的参数；HTTP 和脚本入口共用会话互斥与流关闭逻辑。"""
    with _session_guard:
        busy = thread_id in _active_sessions
        if not busy:
            _active_sessions.add(thread_id)
    if busy:
        yield SESSION_BUSY_REPLY
        return
    try:
        async with aclosing(_get_ai_stream(user_input, thread_id, request_id)) as stream:
            async for chunk in stream:
                yield chunk
    finally:
        with _session_guard:
            _active_sessions.remove(thread_id)


async def _get_ai_stream(user_input: str, thread_id: str, request_id: str | None = None):
    #把会话 id 绑到日志上下文：之后 router / 节点 / retriever 打的所有日志都会带上它，
    #排查某轮对话时直接 grep 这个 id 就能拉出完整链路，不用再拿时间戳去对齐几个文件。
    set_session_id(thread_id)
    usage = TurnUsage(request_id or uuid4().hex)
    budget = TurnBudget(usage.request_id)
    queue = asyncio.Queue(maxsize=1)

    async def produce():
        with budget_scope(budget):
            try:
                async with asyncio.timeout_at(budget.deadline):
                    async with aclosing(_stream_graph(user_input, thread_id, usage)) as stream:
                        async for chunk in stream:
                            await queue.put(chunk)
            except (BudgetStop, TimeoutError) as exc:
                budget.stop(exc.reason if isinstance(exc, BudgetStop) else "deadline")
                await queue.put("\n\n" + BUDGET_REPLY)
            except asyncio.CancelledError:
                budget.stop("cancelled")
                raise
            except Exception as exc:
                await queue.put(exc)
            finally:
                usage.finish()
                budget.finish()
        await queue.put(None)

    producer = asyncio.create_task(produce())
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
    finally:
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer


async def _stream_graph(user_input: str, thread_id: str, usage: TurnUsage):
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [usage]}
    logger.info("turn_start request_id=%s input_chars=%d", usage.request_id, len(user_input))

    checkpoint_lock = _checkpoint_locks.setdefault((asyncio.get_running_loop(), DB_PATH), asyncio.Lock())
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        # SDK的默认锁只保护单个连接，无法协调多会话的建表与写事务。
        checkpointer.lock = checkpoint_lock
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
