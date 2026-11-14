"""单worker真实图/SQLite竞争；依赖替身不访问在线模型或商城。"""
import asyncio
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import budget as budgets
from agent import main_agent as main
from agent import http_api
from agent.budget import BudgetStop, TurnBudget, budget_scope, invoke_model, invoke_query
from agent.prompts import ROUTE_REPLIES
from tests.check_runtime_boundaries import OFFLINE_ORIGIN, offline_http_client


async def collect(query, session, request_id=None):
    async with aclosing(main.stream_reply(query, session, request_id)) as stream:
        return "".join([chunk async for chunk in stream])


def route_model(intent="订单查询", incoming=12):
    return FakeMessagesListChatModel(responses=[AIMessage(
        content=json.dumps({"intent": intent, "subjects": [], "needs_clarification": False}),
        usage_metadata={"input_tokens": incoming, "output_tokens": 3, "total_tokens": incoming + 3})])


class GraphConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_io_excludes_other_connections_without_serializing_models(self):
        both_setup = asyncio.Event()
        model_barrier = asyncio.Barrier(2)
        setup_connections = set()
        active_io = 0
        maximum_io = 0
        original_setup = AsyncSqliteSaver.setup
        original_script = aiosqlite.Connection.executescript

        async def observe_setup(saver):
            setup_connections.add(id(saver.conn))
            if len(setup_connections) == 2:
                both_setup.set()
            await original_setup(saver)

        @asynccontextmanager
        async def gate_script(connection, *args, **kwargs):
            nonlocal active_io, maximum_io
            active_io += 1
            maximum_io = max(maximum_io, active_io)
            try:
                # 第一个事务等待另一个连接已进入setup；旧实现会同时进入此处。
                await both_setup.wait()
                async with original_script(connection, *args, **kwargs) as cursor:
                    yield cursor
            finally:
                active_io -= 1

        model = AsyncMock()

        async def gated(messages, **kwargs):
            await model_barrier.wait()
            return await route_model().ainvoke(messages, config=kwargs.get("config"))

        model.ainvoke.side_effect = gated
        with tempfile.TemporaryDirectory(prefix="agent-g-") as directory, \
                patch.object(main, "DB_PATH", str(Path(directory) / "io.db")), \
                patch.object(main.router, "client", model), \
                patch.object(AsyncSqliteSaver, "setup", observe_setup), \
                patch.object(aiosqlite.Connection, "executescript", gate_script):
            tasks = [asyncio.create_task(collect("checkpoint gate", f"g-io-{i}")) for i in range(2)]
            try:
                replies = await asyncio.wait_for(asyncio.gather(*tasks), 5)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            self.assertEqual(replies, [ROUTE_REPLIES["订单查询"]] * 2)
            self.assertEqual(len(setup_connections), 2)
            self.assertEqual(maximum_io, 1)
            self.assertEqual(active_io, 0)
            self.assertEqual(model.ainvoke.await_count, 2)

    async def test_ten_same_session_cold_start_accepts_one(self):
        start = asyncio.Barrier(10)
        entered, release, rejected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        accepted_queries, rejected_queries = [], []
        model = AsyncMock()

        async def gated(messages, **kwargs):
            accepted_queries.append(messages[-1].content)
            entered.set()
            await release.wait()
            return await route_model().ainvoke(messages, config=kwargs.get("config"))

        async def run(index):
            query = f"same-{index}"
            await start.wait()
            reply = await collect(query, "g-same", f"g-same-{index}")
            if reply == main.SESSION_BUSY_REPLY:
                rejected_queries.append(query)
                if len(rejected_queries) == 9:
                    rejected.set()
            return reply

        model.ainvoke.side_effect = gated
        with tempfile.TemporaryDirectory(prefix="agent-g-") as directory, \
                patch.object(main, "DB_PATH", str(Path(directory) / "cold.db")), \
                patch.object(main.router, "client", model):
            tasks = [asyncio.create_task(run(index)) for index in range(10)]
            try:
                await asyncio.wait_for(entered.wait(), 5)
                await asyncio.wait_for(rejected.wait(), 5)
                self.assertEqual(model.ainvoke.await_count, 1)
                async with AsyncSqliteSaver.from_conn_string(main.DB_PATH) as saver:
                    state = await main.builder.compile(checkpointer=saver).aget_state({"configurable": {"thread_id": "g-same"}})
                    humans = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
                    self.assertEqual(humans, accepted_queries)
            finally:
                release.set()
                replies = await asyncio.wait_for(asyncio.gather(*tasks), 5)
            self.assertEqual(replies.count(main.SESSION_BUSY_REPLY), 9)
            self.assertEqual(replies.count(ROUTE_REPLIES["订单查询"]), 1)
            self.assertNotIn("g-same", main._active_sessions)
            self.assertEqual(await collect("retry", "g-same"), ROUTE_REPLIES["订单查询"])
            async with AsyncSqliteSaver.from_conn_string(main.DB_PATH) as saver:
                state = await main.builder.compile(checkpointer=saver).aget_state({"configurable": {"thread_id": "g-same"}})
                self.assertEqual([m.content for m in state.values["messages"] if isinstance(m, HumanMessage)], accepted_queries)
            self.assertEqual(model.ainvoke.await_count, 2)

    async def test_ten_sessions_twenty_rounds_real_graph_and_cold_sqlite(self):
        # Windows调试栈采集会主导批量图执行耗时；保留屏障、15秒期限和全部正确性断言。
        asyncio.get_running_loop().set_debug(False)
        observed = {}
        barrier = asyncio.Barrier(10)
        model = AsyncMock()

        async def gated(messages, **kwargs):
            current = budgets.CURRENT_BUDGET.get()
            observed[current.request_id] = current
            await barrier.wait()
            index = int(messages[-1].content.split(":")[1])
            return await route_model(incoming=12 + index).ainvoke(messages, config=kwargs.get("config"))

        model.ainvoke.side_effect = gated
        with tempfile.TemporaryDirectory(prefix="agent-g-") as directory, \
                patch.object(main, "DB_PATH", str(Path(directory) / "cold.db")), \
                patch.object(main.router, "client", model), \
                self.assertLogs("agent.budget", level="INFO") as logs:
            for turn in range(20):
                tasks = [asyncio.create_task(collect(f"turn:{index}:{turn}", f"g-user-{index}", f"g-r{turn}-{index}")) for index in range(10)]
                try:
                    replies = await asyncio.wait_for(asyncio.gather(*tasks), 15)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                self.assertEqual(replies, [ROUTE_REPLIES["订单查询"]] * 10)
                self.assertFalse(main._active_sessions)
            async with AsyncSqliteSaver.from_conn_string(main.DB_PATH) as saver:
                graph = main.builder.compile(checkpointer=saver)
                for index in range(10):
                    state = await graph.aget_state({"configurable": {"thread_id": f"g-user-{index}"}})
                    self.assertEqual([m.content for m in state.values["messages"] if isinstance(m, HumanMessage)],
                                     [f"turn:{index}:{turn}" for turn in range(20)])
                    self.assertEqual(len(state.values["messages"]), 40)
            self.assertEqual(model.ainvoke.await_count, 200)
        self.assertEqual(len(observed), 200)
        self.assertEqual(len({id(value) for value in observed.values()}), 200)
        for request_id, budget in observed.items():
            index = int(request_id.rsplit("-", 1)[1])
            self.assertEqual((budget.calls, budget.spent, budget.reserved, budget.unknown, budget.queries), (1, 15 + index, 0, 0, 0))
            self.assertIsNone(budget.stopped)
            self.assertIn(f"budget_finish request_id={request_id} calls=1 queries=0 charged_tokens={15 + index} ", "\n".join(logs.output))

    async def test_graph_model_failure_timeout_and_recovery(self):
        with tempfile.TemporaryDirectory(prefix="agent-g-") as directory, \
                patch.object(main, "DB_PATH", str(Path(directory) / "state.db")):
            model = AsyncMock()
            model.ainvoke.side_effect = RuntimeError("controlled model failure")
            with patch.object(main.router, "client", model), self.assertRaisesRegex(RuntimeError, "controlled"):
                await collect("failure", "g-failure")
            self.assertNotIn("g-failure", main._active_sessions)
            closed = asyncio.Event()

            async def wait(*args, **kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()

            model.ainvoke.side_effect = wait
            with patch.object(main.router, "client", model), patch.object(TurnBudget, "CALL_SECONDS", .03):
                reply = await collect("timeout", "g-failure")
            self.assertIn(budgets.BUDGET_REPLY, reply)
            self.assertTrue(closed.is_set())
            self.assertNotIn("g-failure", main._active_sessions)
            with patch.object(main.router, "client", route_model()):
                self.assertEqual(await collect("recovered", "g-failure"), ROUTE_REPLIES["订单查询"])

    async def test_real_recommend_graph_tool_failure_and_timeout_release_session(self):
        from agent.tool.schemas import FilterParams, DiscoveryPage
        extracted = AsyncMock()
        extracted.ainvoke.return_value = FilterParams(category="耳机")
        extractor = MagicMock()
        extractor.with_structured_output.return_value.with_config.return_value = extracted
        entered, finish = threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        released = asyncio.Event()

        class ObservedSlot(threading.BoundedSemaphore):
            def release(self, n=1):
                super().release(n)
                loop.call_soon_threadsafe(released.set)

        slot = ObservedSlot(1)

        def waiting(request):
            entered.set()
            if not finish.wait(3):
                raise TimeoutError("test worker not released")
            return DiscoveryPage(items=[], hasMore=False, nextCursor=None)

        with tempfile.TemporaryDirectory(prefix="agent-g-") as directory, \
                patch.object(main, "DB_PATH", str(Path(directory) / "tools.db")), \
                patch.object(main.router, "client", route_model("商品推荐")), \
                patch("agent.nodes.recommend.get_extract_client", return_value=extractor), \
                patch("agent.tool.mall.list_categories", new=AsyncMock(return_value=[])), \
                patch.object(budgets, "_QUERY_SLOTS", slot):
            with patch("agent.tool.mall.search_products", side_effect=RuntimeError("controlled query failure")), \
                    self.assertRaisesRegex(RuntimeError, "query failure"):
                await collect("推荐耳机", "g-tool")
            self.assertNotIn("g-tool", main._active_sessions)
            released.clear()
            with patch("agent.tool.mall.search_products", side_effect=waiting), patch.object(TurnBudget, "QUERY_SECONDS", .03):
                try:
                    reply = await collect("推荐耳机", "g-tool")
                    self.assertIn(budgets.BUDGET_REPLY, reply)
                    self.assertTrue(entered.is_set())
                    self.assertNotIn("g-tool", main._active_sessions)
                    self.assertFalse(slot.acquire(blocking=False))
                finally:
                    finish.set()
                    await asyncio.wait_for(released.wait(), 2)
            with patch("agent.tool.mall.search_products", return_value=DiscoveryPage(items=[], hasMore=False, nextCursor=None)):
                reply = await collect("推荐耳机", "g-tool")
                self.assertIn("没有找到", reply)
                self.assertNotIn(budgets.BUDGET_REPLY, reply)
                self.assertNotIn("g-tool", main._active_sessions)

    async def test_sse_disconnect_cancels_real_graph_model_and_releases_session(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        model = AsyncMock()

        async def wait(*args, **kwargs):
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        model.ainvoke.side_effect = wait
        async with AsyncExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="agent-g-"))
            db = Path(directory) / "stream.db"
            stack.enter_context(patch.object(main, "DB_PATH", str(db)))
            stack.enter_context(patch.object(main.router, "client", model))
            client = await stack.enter_async_context(offline_http_client(db))
            response = await client.post("/api/sessions")
            token = response.raise_for_status().json()["session_token"]
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
                     "method": "POST", "scheme": "http", "path": "/api/chat", "raw_path": b"/api/chat",
                     "query_string": b"", "root_path": "", "server": (client.base_url.host, 80), "client": ("127.0.0.1", 1234),
                     "headers": [(b"origin", OFFLINE_ORIGIN.encode()), (b"content-type", b"application/json"),
                                 (b"authorization", ("Bearer " + token).encode())]}
            first = True
            sent = []

            async def receive():
                nonlocal first
                if first:
                    first = False
                    return {"type": "http.request", "body": b'{"message":"disconnect"}', "more_body": False}
                await entered.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            await asyncio.wait_for(http_api.app(scope, receive, send), 5)
            self.assertEqual(sent[0]["status"], 200)
            self.assertTrue(closed.is_set())
            self.assertNotIn(token.split(".")[0], main._active_sessions)
            self.assertEqual(model.ainvoke.await_count, 1)
            payload = b"".join(message.get("body", b"") for message in sent)
            self.assertNotIn(b"event: done", payload)
            self.assertNotIn(b"event: error", payload)
            with patch.object(main.router, "client", route_model()):
                self.assertEqual(await collect("after-disconnect", token.split(".")[0]), ROUTE_REPLIES["订单查询"])


class QueryResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_cancel_keeps_slot_until_worker_really_exits(self):
        loop = asyncio.get_running_loop()
        entered, released = asyncio.Event(), asyncio.Event()
        finish = threading.Event()

        class ObservedSlot(threading.BoundedSemaphore):
            def release(self, n=1):
                super().release(n)
                loop.call_soon_threadsafe(released.set)

        slot = ObservedSlot(1)

        def query():
            loop.call_soon_threadsafe(entered.set)
            if not finish.wait(3):
                raise TimeoutError("test did not release worker")
            return "late-result"

        with patch.object(budgets, "_QUERY_SLOTS", slot), budget_scope(TurnBudget("g-sync-cancel")):
            task = asyncio.create_task(invoke_query(query))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(slot.acquire(blocking=False))
                with budget_scope(TurnBudget("g-other")), self.assertRaisesRegex(BudgetStop, "query_busy"):
                    await invoke_query(lambda: "must not start")
            finally:
                finish.set()
                await asyncio.wait_for(released.wait(), 2)
            with budget_scope(TurnBudget("g-after")):
                self.assertEqual(await invoke_query(lambda: "recovered"), "recovered")
            self.assertTrue(slot.acquire(blocking=False))
            slot.release()

    async def test_async_cancel_timeout_and_failure_release_slot(self):
        for mode in ("cancel", "timeout", "failure"):
            with self.subTest(mode=mode):
                entered, closed = asyncio.Event(), asyncio.Event()
                slot = threading.BoundedSemaphore(1)
                budget = TurnBudget("g-async-" + mode)
                budget.QUERY_SECONDS = .03 if mode == "timeout" else 5

                async def query():
                    try:
                        entered.set()
                        if mode == "failure":
                            raise RuntimeError("controlled tool error")
                        await asyncio.Event().wait()
                    finally:
                        closed.set()

                with patch.object(budgets, "_QUERY_SLOTS", slot), budget_scope(budget):
                    task = asyncio.create_task(invoke_query(query))
                    await asyncio.wait_for(entered.wait(), 1)
                    if mode == "cancel":
                        task.cancel()
                        expected = asyncio.CancelledError
                    else:
                        expected = BudgetStop if mode == "timeout" else RuntimeError
                    with self.assertRaises(expected):
                        await task
                    self.assertTrue(closed.is_set())
                    with budget_scope(TurnBudget("g-probe")):
                        self.assertEqual(await invoke_query(lambda: "reused"), "reused")
                    self.assertTrue(slot.acquire(blocking=False))
                    slot.release()

    async def test_sync_failure_releases_slot_and_model_exception_settles_budget(self):
        def failed():
            raise RuntimeError("controlled sync error")

        slot = threading.BoundedSemaphore(1)
        budget = TurnBudget("g-failure-settle")
        with patch.object(budgets, "_QUERY_SLOTS", slot), budget_scope(budget):
            with self.assertRaisesRegex(RuntimeError, "sync error"):
                await invoke_query(failed)
            self.assertEqual(await invoke_query(lambda: "ok"), "ok")
            model = AsyncMock()
            model.ainvoke.side_effect = RuntimeError("controlled model error")
            with self.assertRaisesRegex(RuntimeError, "model error"):
                await invoke_model(model, [HumanMessage(content="hello")], purpose="answer")
            self.assertEqual(budget.reserved, 0)
            self.assertEqual(budget.unknown, 1)
            self.assertGreater(budget.spent, 0)
        self.assertTrue(slot.acquire(blocking=False))
        slot.release()
