import asyncio
import json
from contextlib import aclosing
import threading
import unittest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from agent.budget import TurnBudget, BudgetStop, BUDGET_REPLY, budget_scope, invoke_model, invoke_query, estimate_input
from agent.memory import MemoryManager
from agent.tool.schemas import FilterParams
from test_memory import conversation
from check_runtime_boundaries import app, collect

MESSAGES = [HumanMessage(content="测试")]


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_graph_shares_budget_and_usage_callback(self):
        response = AIMessage(content=json.dumps({"intent": "订单查询", "subjects": [], "needs_clarification": False}),
                             usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
        model = FakeMessagesListChatModel(responses=[response])
        with app(":memory:") as (main, _, _), patch.object(main.router, "client", model):
            with self.assertLogs("agent.budget", level="INFO") as logs, \
                    self.assertLogs("agent.telemetry", level="INFO") as telemetry:
                reply = await collect(main, "我的订单到哪了", "budget-graph")
            self.assertTrue(reply)
            self.assertNotIn(BUDGET_REPLY, reply)
            text = "".join(logs.output)
            self.assertIn("charged_tokens=15 estimated=False", text)
            self.assertIn("calls=1 queries=0 charged_tokens=15 unknown_usage=0", text)
            self.assertIn("calls=1 input_tokens=12 output_tokens=3", "".join(telemetry.output))
            self.assertFalse(main._active_sessions)

    async def test_real_graph_budget_stop_not_swallowed_by_router(self):
        model = AsyncMock()
        with app(":memory:") as (main, _, _), patch.object(main.router, "client", model), \
                patch.object(TurnBudget, "MAX_CALLS", 0):
            reply = await collect(main, "我的订单到哪了", "budget-blocked")
            self.assertIn(BUDGET_REPLY, reply)
            model.ainvoke.assert_not_called()
            self.assertFalse(main._active_sessions)

    async def test_query_attempt_limit_before_execution(self):
        calls = []
        def query():
            calls.append(1)
            return "ok"
        budget = TurnBudget("queries")
        with budget_scope(budget):
            for _ in range(6):
                self.assertEqual(await invoke_query(query), "ok")
            with self.assertRaisesRegex(BudgetStop, "query_calls"):
                await invoke_query(query)
        self.assertEqual(len(calls), 6)

    async def test_calls_capped_before_network_and_unknown_charged(self):
        budget = TurnBudget("calls")
        model = AsyncMock()
        model.ainvoke.return_value = AIMessage(content="ok")
        with budget_scope(budget):
            for _ in range(8):
                await invoke_model(model, MESSAGES, purpose="answer", max_output=20)
            with self.assertRaisesRegex(BudgetStop, "model_calls"):
                await invoke_model(model, MESSAGES, purpose="answer", max_output=20)
        self.assertEqual(model.ainvoke.await_count, 8)
        self.assertEqual(budget.spent, 8 * (estimate_input(MESSAGES) + 20))
        self.assertEqual(budget.unknown, 8)
        self.assertEqual(budget.reserved, 0)

    async def test_atomic_token_reservations(self):
        budget = TurnBudget("parallel")
        budget.MAX_TOKENS = estimate_input(MESSAGES) + 20
        entered, release = asyncio.Event(), asyncio.Event()
        model = AsyncMock()
        async def waiting(*args, **kwargs):
            entered.set()
            await release.wait()
            return AIMessage(content="ok")
        model.ainvoke.side_effect = waiting
        with budget_scope(budget):
            first = asyncio.create_task(invoke_model(model, MESSAGES, purpose="answer", max_output=20))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                with self.assertRaisesRegex(BudgetStop, "tokens"):
                    await invoke_model(model, MESSAGES, purpose="answer", max_output=20)
            finally:
                release.set()
                await first
        self.assertEqual(model.ainvoke.await_count, 1)
        self.assertEqual(budget.spent, budget.MAX_TOKENS)

    async def test_usage_settles_real_callback_and_isolation(self):
        model = FakeMessagesListChatModel(responses=[AIMessage(content="ok", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})])
        async def run(name):
            budget = TurnBudget(name)
            with budget_scope(budget):
                await invoke_model(model, MESSAGES, purpose="answer")
            return budget
        a, b = await asyncio.gather(run("a"), run("b"))
        self.assertEqual((a.calls, b.calls, a.spent, b.spent), (1, 1, 15, 15))
        self.assertEqual((a.unknown, a.reserved), (0, 0))

    async def test_input_schema_budget_and_answer_reserve(self):
        model = AsyncMock()
        budget = TurnBudget("input")
        budget.MAX_INPUT = estimate_input(MESSAGES)
        with budget_scope(budget), self.assertRaisesRegex(BudgetStop, "single_call_tokens"):
            await invoke_model(model, MESSAGES, schema=FilterParams)
        model.ainvoke.assert_not_called()
        budget = TurnBudget("reserve")
        budget.MAX_TOKENS = 6000
        with budget_scope(budget), self.assertRaisesRegex(BudgetStop, "tokens"):
            await invoke_model(model, MESSAGES)
        model.ainvoke.assert_not_called()

    async def test_one_repair_shared_and_failures_still_charged(self):
        budget = TurnBudget("repair")
        budget.answer_started = True
        model = AsyncMock()
        model.ainvoke.side_effect = RuntimeError("private")
        with budget_scope(budget):
            with self.assertRaises(RuntimeError):
                await invoke_model(model, MESSAGES, purpose="repair")
            with self.assertRaisesRegex(BudgetStop, "repair_calls"):
                await invoke_model(model, MESSAGES, purpose="repair")
        self.assertEqual(budget.unknown, 1)
        self.assertGreater(budget.spent, 0)
        self.assertEqual(model.ainvoke.await_count, 1)

    async def test_model_timeout_cancels_and_blocks_next_call(self):
        closed = asyncio.Event()
        async def wait(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        model = AsyncMock()
        model.ainvoke.side_effect = wait
        budget = TurnBudget("timeout")
        budget.CALL_SECONDS = 0.01
        with budget_scope(budget):
            with self.assertRaisesRegex(BudgetStop, "model_timeout"):
                await invoke_model(model, MESSAGES, purpose="answer")
            with self.assertRaises(BudgetStop):
                await invoke_model(model, MESSAGES, purpose="answer")
        self.assertTrue(closed.is_set())
        self.assertEqual(model.ainvoke.await_count, 1)
        self.assertEqual(budget.reserved, 0)

    async def test_summary_skips_without_deleting_when_budget_exhausted(self):
        model = AsyncMock()
        budget = TurnBudget("summary")
        budget.calls = 8
        manager = MemoryManager(model)
        state = {"messages": conversation(6), "summary": "原摘要"}
        with budget_scope(budget):
            self.assertEqual(await manager.summarize_node(state), {})
        model.ainvoke.assert_not_called()
        self.assertEqual(len(state["messages"]), 12)
        self.assertEqual(state["summary"], "原摘要")

    async def test_query_timeout_keeps_background_slot_until_done(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        slot = threading.BoundedSemaphore(1)
        def query():
            entered.set()
            try:
                release.wait(2)
                return "late"
            finally:
                closed.set()
        budget = TurnBudget("query")
        budget.QUERY_SECONDS = 0.03
        try:
            with patch("agent.budget._QUERY_SLOTS", slot), budget_scope(budget):
                with self.assertRaisesRegex(BudgetStop, "query_timeout"):
                    await invoke_query(query)
                self.assertTrue(entered.is_set())
                self.assertFalse(slot.acquire(blocking=False))
                with budget_scope(TurnBudget("other")), self.assertRaisesRegex(BudgetStop, "query_busy"):
                    await invoke_query(lambda: "not sent")
                release.set()
                await asyncio.to_thread(closed.wait, 1)
                # work finally releases the slot immediately after the function returns.
                await asyncio.sleep(0)
        finally:
            release.set()

    async def test_real_entry_deadline_and_cancel_release_session(self):
        with app(":memory:") as (main, _, _):
            closed = asyncio.Event()
            async def stream(*args):
                try:
                    await asyncio.Event().wait()
                    yield "never"
                finally:
                    closed.set()
            with patch.object(main, "_stream_graph", stream), patch.object(TurnBudget, "TURN_SECONDS", 0.02):
                reply = await collect(main, "测试", "deadline")
                self.assertIn(BUDGET_REPLY, reply)
                self.assertTrue(closed.is_set())
                self.assertFalse(main._active_sessions)
            closed.clear()
            with patch.object(main, "_stream_graph", stream):
                task = asyncio.create_task(collect(main, "测试", "cancel"))
                await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(closed.is_set())
                self.assertFalse(main._active_sessions)

    async def test_consumer_early_close_stops_producer(self):
        with app(":memory:") as (main, _, _):
            closed = asyncio.Event()
            async def stream(*args):
                try:
                    yield "first"
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            with patch.object(main, "_stream_graph", stream):
                async with aclosing(main.get_ai_stream("测试", "early")) as output:
                    self.assertEqual(await anext(output), "first")
                self.assertTrue(closed.is_set())
                self.assertFalse(main._active_sessions)
