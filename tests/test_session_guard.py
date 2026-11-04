import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from check_runtime_boundaries import app, collect, race


class SessionGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_graph_same_and_different_sessions(self):
        fd, path = tempfile.mkstemp(suffix=".db", dir=Path(__file__).parent)
        os.close(fd)
        try:
            with app(path) as (main, _, _):
                async with AsyncSqliteSaver.from_conn_string(path) as saver:
                    await saver.setup()
                for index, same in enumerate((True, False)):
                    result = await race(main, path, same, index)
                    self.assertTrue(result["passed"], result)
                    self.assertFalse(main._active_sessions)
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_error_releases_admission(self):
        with app(":memory:") as (main, _, _):
            async def failed(*args):
                raise RuntimeError("test failure")
                yield  # async generator, same contract as the real stream

            with patch.object(main, "_get_ai_stream", failed):
                for _ in range(2):
                    with self.assertRaises(RuntimeError):
                        await collect(main, "你好", "same")
                    self.assertFalse(main._active_sessions)

    async def test_cancel_releases_admission_and_closes_inner_stream(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        with app(":memory:") as (main, _, _):
            async def waiting(*args):
                try:
                    entered.set()
                    await asyncio.Event().wait()
                    yield "answer"
                finally:
                    closed.set()

            with patch.object(main, "_get_ai_stream", waiting):
                task = asyncio.create_task(collect(main, "你好", "same"))
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(closed.is_set())
                self.assertFalse(main._active_sessions)

    async def test_early_close_and_cross_thread_admission(self):
        closed = asyncio.Event()
        with app(":memory:") as (main, _, _):
            async def chunks(*args):
                try:
                    yield "first"
                    yield "second"
                finally:
                    closed.set()

            with patch.object(main, "_get_ai_stream", chunks):
                stream = main.get_ai_stream("你好", "same")
                self.assertEqual(await anext(stream), "first")
                try:
                    # Another OS thread and event loop still shares the same admission guard.
                    reply = await asyncio.to_thread(lambda: asyncio.run(collect(main, "另一个请求", "same")))
                    self.assertEqual(reply, main.SESSION_BUSY_REPLY)
                    self.assertEqual(await collect(main, "其他会话", "other"), "firstsecond")
                    closed.clear()
                finally:
                    await stream.aclose()
                self.assertTrue(closed.is_set())
                self.assertFalse(main._active_sessions)
                self.assertEqual(await collect(main, "重试", "same"), "firstsecond")


if __name__ == "__main__":
    unittest.main()
