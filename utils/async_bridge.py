"""让同步代码（Streamlit 脚本）安全地驱动 async generator。

🔴 为什么不再用 nest_asyncio
================================
nest_asyncio 为了让事件循环「可重入」，把 loop._run_once 改成了这样
（nest_asyncio.py:130，执行每个 handle 前后动了 current task）：

    curr_task = curr_tasks.pop(self, None)   # 先把当前 task 摘掉
    try:
        handle._run()                        # 真正跑这一步
    finally:
        curd_tasks[self] = curr_task         # 再放回去

而 Python 3.12 起，`asyncio.wait_for` / `asyncio.timeout` 内部会做
`async with timeouts.timeout(...)`，它要求「当前必须在一个 Task 里」
（`tasks.current_task()` 不能为 None）。两者一撞，直接抛：

    RuntimeError: Timeout should be used inside a task

本项目跑在 Python 3.14，langgraph 的 _runner.py 里就有 `asyncio.wait_for`，
所以只要 nest_asyncio.apply() 过，流式输出必崩。已把它从项目里彻底移除。

🔧 替代方案
============
**一个后台线程常驻一个事件循环**，主线程只做同步等待 + 取数据：

1. 主线程压根没有 running loop，不存在嵌套问题，`asyncio.timeout` 正常工作；
2. 循环长期复用，AsyncSqliteSaver、HTTP 连接池不用每轮重建；
3. 流式照样流式——chunk 通过 queue 一个个送回主线程。

用法：
    from utils.async_bridge import iter_async, run_async

    for chunk in iter_async(get_ai_stream, "你好"):   # 流式
        print(chunk, end="")

    result = run_async(some_coro())                   # 只要最终结果
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Callable

_SENTINEL = object()  # 队列里的「结束」标记，不能用 None（chunk 本身可能是 None）

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """后台那个常驻事件循环（第一次用到时才启动）。

    用 daemon 线程：主进程退出时它自动跟着走，不会拦着退出。
    """
    global _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever,
                name="agent-event-loop",
                daemon=True,
            ).start()
            _loop = loop
        return _loop


def run_async(coro) -> Any:
    """在后台循环里跑一个协程，同步等它出结果（异常原样抛回主线程）。"""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()


def iter_async(factory: Callable[..., Any], *args, **kwargs):
    """把 async generator 函数转成同步迭代器。

    :param factory: async generator 函数本身（不是调用后的对象），
        例如 `iter_async(get_ai_stream, "你好", thread_id="u1")`。
        ⚠️ 传函数而不是 `get_ai_stream("你好")` 是有意的：
        async generator 由哪个线程/循环驱动，就在哪个线程里创建，
        不要让协程对象跨线程传来传去。
    """
    chunk_queue: queue.Queue = queue.Queue()

    async def _drive() -> None:
        try:
            async for chunk in factory(*args, **kwargs):
                chunk_queue.put((True, chunk))
        except BaseException as exc:  # noqa: BLE001 - 要原样抛回主线程
            chunk_queue.put((False, exc))
        else:
            chunk_queue.put((False, _SENTINEL))

    future = asyncio.run_coroutine_threadsafe(_drive(), _get_loop())

    try:
        while True:
            is_chunk, payload = chunk_queue.get()
            if is_chunk:
                yield payload
                continue
            if payload is _SENTINEL:
                return
            raise payload  # 后台协程抛的异常，在主线程重新抛出
    finally:
        # 调用方提前 break / 页面 rerun 时，别把后台协程扔在那儿空跑
        if not future.done():
            future.cancel()
        elif not future.cancelled():
            future.result()  # 兜底：把后台协程里的异常带回来
