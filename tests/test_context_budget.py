from tests.mall_fixtures import selection
import json
import unittest
from unittest.mock import AsyncMock,patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from agent.memory import MemoryManager
from agent.state import AgentState
from agent.tool.schemas import FilterParams
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.graph import START, END, StateGraph

from tests.test_memory import conversation
from tests.check_runtime_boundaries import app
from agent.memory import MemoryManager
from agent.state import AgentState
from agent.router import IntentRouter
from agent.telemetry import TurnUsage


class ContextBudgetTests(unittest.IsolatedAsyncioTestCase):

    async def test_recommend_query_replaces_previous_budget(self):
        with app(":memory:"):
            from agent.nodes.recommend import recommend_node

            state = {
                "messages": [
                    HumanMessage(content="推荐300元以内的耳机"),
                    AIMessage(content="你要什么颜色？"),
                    HumanMessage(content="黑色，预算提高到500元"),
                ],
                "intent": "商品推荐",
            }
            # 模拟提取结果，验证节点怎样使用结果；不验证真实模型的理解能力。
            extracted = FilterParams(category="耳机", max_price=500, color="黑色")
            with (
                patch("agent.nodes.recommend.invoke_model", new_callable=AsyncMock, return_value=extracted) as model,
                patch("agent.nodes.recommend.find_products", new_callable=AsyncMock, return_value=selection()) as query,
            ):
                update = await recommend_node(state)

            model.assert_awaited_once()
            query.assert_awaited_once()
            sent_text = "\n".join(message.content for message in model.await_args.args[1])
            self.assertIn("推荐300元以内的耳机", sent_text)
            self.assertIn("黑色，预算提高到500元", sent_text)

            search_filters = query.await_args.args[1].model_dump()
            self.assertEqual(search_filters["category"], "耳机")
            self.assertEqual(search_filters["color"], "黑色")
            self.assertEqual(search_filters["max_price"], 500)
            self.assertEqual(update["task_context"]["filters"], extracted.model_dump(mode="json", exclude={"next_page"}))

    async def test_recommend_uses_saved_conditions(self):
        with app(":memory:"):
            from agent.nodes.recommend import recommend_node

            state = {
                "messages": [HumanMessage(content="黑色，预算提高到500元")],
                "intent": "商品推荐",
                "task_context": {
                    "intent": "商品推荐",
                    "filters": {"category": "耳机", "min_price": None, "max_price": 300, "color": None},
                },
            }
            # 模拟提取结果，验证节点怎样使用结果；不验证真实模型的理解能力。
            extracted = FilterParams(category="耳机", max_price=500, color="黑色")
            with (
                patch("agent.nodes.recommend.invoke_model", new_callable=AsyncMock, return_value=extracted) as model,
                patch("agent.nodes.recommend.find_products", new_callable=AsyncMock, return_value=selection()) as query,
            ):
                update = await recommend_node(state)

            model.assert_awaited_once()
            query.assert_awaited_once()
            sent_text = "\n".join(message.content for message in model.await_args.args[1])
            task = json.loads(next(m.content.split("\n", 1)[1] for m in model.await_args.args[1]
                                   if m.content.startswith("【结构化业务状态")))
            self.assertEqual(task["filters"]["category"], "耳机")
            self.assertEqual(task["filters"]["max_price"], 300)
            self.assertIn("黑色，预算提高到500元", sent_text)

            search_filters = query.await_args.args[1].model_dump()
            self.assertEqual(search_filters["category"], "耳机")
            self.assertEqual(search_filters["color"], "黑色")
            self.assertEqual(search_filters["max_price"], 500)
            self.assertEqual(update["task_context"]["filters"], extracted.model_dump(mode="json", exclude={"next_page"}))

    async def test_recommend_clears_cancelled_color(self):
        with app(":memory:"):
            from agent.nodes.recommend import recommend_node

            state = {
                "messages": [
                    HumanMessage(content="颜色不限"),
                ],
                "intent": "商品推荐",
                "task_context": {
                    "intent": "商品推荐",
                    "filters": {
                        "category": "耳机",
                        "min_price": None,
                        "max_price": 500,
                        "color": "黑色",
                    },
                },
            }

            #假设模型已经取出了正确的新条件
            extracted=FilterParams(
                max_price=500,
                category="耳机",
                color=None,
            )

            with(
                patch(
                    "agent.nodes.recommend.invoke_model",
                    new_callable=AsyncMock,
                    return_value=extracted,
                )as model,
                patch(
                    "agent.nodes.recommend.find_products",
                    new_callable=AsyncMock,
                    return_value=selection(),
                )as query,
            ):
                update = await recommend_node(state)
            model.assert_awaited_once()

            sent_messages = model.await_args.args[1]
            sent_text = "\n".join(message.content for message in sent_messages)

            # 模型收到旧颜色和取消要求
            task = json.loads(next(m.content.split("\n", 1)[1] for m in sent_messages
                                   if m.content.startswith("【结构化业务状态")))
            self.assertEqual(task["filters"]["color"], "黑色")
            self.assertIn("颜色不限", sent_text)

            search_filters = query.await_args.args[1].model_dump()

            # 查询不再限制颜色，品类和预算仍然保留
            self.assertIsNone(search_filters["color"])
            self.assertEqual(search_filters["category"], "耳机")
            self.assertEqual(search_filters["max_price"], 500)

            # 保存下来的条件明确记录：颜色限制已取消
            saved_filters = update["task_context"]["filters"]
            self.assertIsNone(saved_filters["color"])
            self.assertEqual(saved_filters["max_price"], "500")

    async def test_route_is_smaller_and_current_question_is_whole(self):
        memory = MemoryManager()
        state = {"messages": conversation(5) + [HumanMessage(content="便宜点")], "summary": "预算500"}
        route = memory.build_messages(state, "规则", purpose="route")
        answer = memory.build_messages(state, "规则")
        self.assertLess(sum(len(m.content) for m in route), sum(len(m.content) for m in answer))
        self.assertEqual(route[-1].content, "便宜点")
        state["messages"].append(HumanMessage(content="长" * 6000))
        self.assertEqual(memory.build_messages(state, "规则", purpose="route")[-1].content, "长" * 6000)

    async def test_summary_batches_without_losing_deferred_turn(self):
        model = AsyncMock()
        model.ainvoke.return_value = AIMessage(content="偏好黑色")
        memory = MemoryManager(model)
        state = {"messages": conversation(5)}
        self.assertEqual(await memory.summarize_node(state), {})
        model.ainvoke.assert_not_called()
        self.assertIn("用户需求0", [m.content for m in memory.build_messages(state, "规则")])
        update = await memory.summarize_node({"messages": conversation(6)})
        self.assertEqual(len(update["messages"]), 4)
        self.assertEqual(model.ainvoke.await_count, 1)

    async def test_task_switch_and_clarification(self):
        model = AsyncMock()
        router = IntentRouter(model)
        state = {"messages": [HumanMessage(content="查订单")], "intent": "澄清",
                 "task_context": {"intent": "概览筛选", "filters": {"max_price": 500}}}
        for intent, clear in (("订单查询", True), ("概览筛选", False), ("澄清", False)):
            model.ainvoke.return_value = AIMessage(content=json.dumps({"intent": intent, "subjects": [], "needs_clarification": False}))
            update = await router.node(state)
            self.assertEqual("task_context" in update, clear)
        isolated = MemoryManager().build_messages({**state, "intent": "订单查询"}, "规则")
        self.assertFalse(any("max_price" in m.content for m in isolated))

    async def test_filter_state_replaced_and_cancelled_not_merged(self):
        from agent.tool.schemas import FilterParams
        with app(":memory:") as (_, deps, products):
            from agent.nodes.total import total_node
            extractor = AsyncMock()
            deps.extract_client.with_structured_output.return_value.with_config.return_value = extractor
            products.find_products.return_value = selection()
            state = {"messages": [HumanMessage(content="改成300，颜色不限")], "intent": "概览筛选",
                     "task_context": {"intent": "概览筛选", "filters": {"category": "耳机", "max_price": 500, "color": "黑色"}}}
            extractor.ainvoke.return_value = FilterParams(category="耳机", max_price=300, color=None)
            update = await total_node(state)
            self.assertEqual(update["task_context"]["filters"]["max_price"], "300")
            self.assertIsNone(update["task_context"]["filters"]["color"])
            payload = "".join(m.content for m in extractor.ainvoke.call_args.args[0])
            task = json.loads(next(m.content.split("\n", 1)[1] for m in extractor.ainvoke.call_args.args[0]
                                   if m.content.startswith("【结构化业务状态")))
            self.assertEqual(task["filters"]["max_price"], 500)
            self.assertIn("改成300，颜色不限", payload)

    async def test_filter_failure_preserves_state_and_zero_is_a_condition(self):
        from agent.tool.schemas import FilterParams
        with app(":memory:") as (_, deps, products):
            from agent.nodes.total import total_node
            extractor = AsyncMock()
            deps.extract_client.with_structured_output.return_value.with_config.return_value = extractor
            state = {"messages": [HumanMessage(content="免费")], "task_context": {"filters": {"color": "黑色"}}}
            extractor.ainvoke.side_effect = TimeoutError()
            update = await total_node(state)
            self.assertNotIn("task_context", update)
            products.find_products.assert_not_called()
            extractor.ainvoke.side_effect = RuntimeError("bug")
            with self.assertRaises(RuntimeError):
                await total_node(state)
            extractor.ainvoke.side_effect = None
            extractor.ainvoke.return_value = FilterParams(max_price=0)
            products.find_products.return_value = selection()
            await total_node(state)
            self.assertEqual(products.find_products.call_args.args[1].max_price, 0)

    async def test_usage_callback_inherited_by_graph_model(self):
        model = FakeMessagesListChatModel(responses=[AIMessage(content="private-answer",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
                            "output_token_details": {"reasoning": 1}})])
        async def answer(state):
            return {"messages": [await model.ainvoke(state["messages"])]}
        builder = StateGraph(AgentState)
        builder.add_node("answer", answer)
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        usage = TurnUsage("test-request")
        with self.assertLogs("agent.telemetry", level="INFO") as logs:
            await builder.compile().ainvoke({"messages": [HumanMessage(content="private-input")]}, config={"callbacks": [usage]})
            usage.finish()
        self.assertEqual((usage.calls, usage.input_tokens, usage.output_tokens), (1, 12, 3))
        text = "".join(logs.output)
        self.assertIn("stage=answer", text)
        self.assertIn("reasoning_tokens=1 finish_reason=stop", text)
        self.assertNotIn("private-", text)

    async def test_unknown_usage_and_error_do_not_log_content(self):
        usage = TurnUsage("test")
        with self.assertLogs("agent.telemetry", level="INFO") as logs:
            for error in (False, True):
                run_id = uuid4()
                usage.on_chat_model_start({}, [[HumanMessage(content="secret")]], run_id=run_id)
                if error:
                    usage.on_llm_error(RuntimeError("secret-key"), run_id=run_id)
                else:
                    usage.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="secret-answer"))]]), run_id=run_id)
            usage.finish()
        self.assertEqual(usage.unknown_usage, 2)
        self.assertEqual(usage.failures, 1)
        self.assertNotIn("secret", "".join(logs.output))
