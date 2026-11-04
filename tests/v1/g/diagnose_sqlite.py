"""只读诊断运行时行为：仅写独立临时库，不改SDK或业务实现。"""
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import tempfile
import sqlite3

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def probe(width, reuse, warm):
    with tempfile.TemporaryDirectory(prefix="g-saver-probe-") as directory:
        path = str(Path(directory) / "probe.db")
        if warm:
            async with AsyncSqliteSaver.from_conn_string(path) as initial:
                await initial.setup()
        async with AsyncExitStack() as lifetime:
            shared = [await lifetime.enter_async_context(AsyncSqliteSaver.from_conn_string(path)) for _ in range(width)] if reuse else []
            for turn in range(20):
                barrier = asyncio.Barrier(width)

                async def worker(index):
                    phase = "open"
                    async with AsyncExitStack() as per_turn:
                        saver = shared[index] if reuse else await per_turn.enter_async_context(AsyncSqliteSaver.from_conn_string(path))
                        try:
                            config = {"configurable": {"thread_id": f"probe-{index}", "checkpoint_ns": ""}}
                            phase = "setup"
                            await saver.setup()
                            phase = "get"
                            await saver.aget_tuple(config)
                            await barrier.wait()
                            checkpoint = empty_checkpoint()
                            checkpoint["channel_values"] = {"messages": [f"turn-{x}" for x in range(turn + 1)]}
                            phase = "put"
                            updated = await saver.aput(config, checkpoint, {}, {})
                            phase = "writes"
                            await saver.aput_writes(updated, [("channel", "value")], "task")
                            return None
                        except (sqlite3.Error, asyncio.BrokenBarrierError) as exc:
                            # 收集SQLite故障并解除同批屏障；不改变被测SDK行为。
                            await barrier.abort()
                            return {"phase": phase, "error": type(exc).__name__, "sqlite_code": getattr(exc, "sqlite_errorcode", None), "sqlite_name": getattr(exc, "sqlite_errorname", None)}

                results = await asyncio.wait_for(asyncio.gather(*(worker(i) for i in range(width))), 20)
                failures = [result for result in results if result]
                if failures:
                    return {"width": width, "reuse": reuse, "warm": warm, "turn": turn, "failures": failures}
            return {"width": width, "reuse": reuse, "warm": warm, "rounds": 20, "failures": []}


async def main():
    for warm in (False, True):
        for width, reuse in ((2, False), (10, False), (2, True), (10, True)):
            print(json.dumps(await probe(width, reuse, warm), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
