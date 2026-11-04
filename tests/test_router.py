import asyncio
import json
from pathlib import Path
import sys
import importlib
import logging
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from agent.state import AgentState
from agent.nodes.routing import route_reply_node
from agent.prompts import ROUTE_REPLIES, ROUTER_SYSTEM
from agent.router import IntentRouter, ROUTE_MAP, ROUTE_NAMES, route_with_fallback


def route_response(intent, subjects=None, needs_clarification=False):
    return AIMessage(content=json.dumps({"intent": intent, "subjects": subjects if subjects is not None else ["A", "B"],
                                         "needs_clarification": needs_clarification}, ensure_ascii=False))


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_id_normalization_preserves_evidence_and_ambiguity_checks(self):
        cases = [
            ("商品编号101红色有货吗", "业务咨询", ["商品编号101红色"], ("业务咨询", ["101"])),
            ("商品编号101红色有货吗", "业务咨询", ["商品编号999红色"], ("澄清", [])),
            ("型号101红色有货吗", "业务咨询", ["型号101红色"], ("业务咨询", ["型号101红色"])),
            ("商品编号101和102", "业务咨询", ["商品编号101和102"], ("业务咨询", ["商品编号101和102"])),
            ("对比商品编号101红色和商品编号101蓝色", "商品对比", ["商品编号101红色", "商品编号101蓝色"], ("澄清", [])),
        ]
        for query, intent, subjects, expected in cases:
            with self.subTest(query=query, subjects=subjects):
                self.client.ainvoke.return_value = route_response(intent, subjects)
                result = await self.router.decision({"messages": [HumanMessage(content=query)]})
                self.assertEqual(result, expected)

    async def test_arrays_are_rejected_without_relaxing_validation(self):
        valid = {"intent": "业务咨询", "subjects": ["A"], "needs_clarification": False}
        self.client.ainvoke.return_value = AIMessage(content=json.dumps([valid]))
        self.assertEqual(await self.router.router(self.state), "识别失败")
        for payload in ([], [valid, valid], ["业务咨询"], [{**valid, "extra": True}],
                        [{**valid, "subjects": ["不存在的商品"]}], [{**valid, "intent": "任意执行"}]):
            self.client.ainvoke.return_value = AIMessage(content=json.dumps(payload))
            self.assertIn(await self.router.router(self.state), ("识别失败", "澄清"))

    async def test_router_passes_configured_non_thinking_option(self):
        client = AsyncMock()
        client.ainvoke.return_value = route_response("澄清", [], True)
        config = {"llm.extra_body": {"enable_thinking": False}}
        with patch("agent.deps.APP_CFG.get", side_effect=lambda key, default=None: config.get(key, default)), \
                patch("agent.deps.os.getenv", return_value="test"), \
                patch("agent.deps.init_chat_model", return_value=client) as build:
            self.assertEqual(await IntentRouter().router({"messages": [HumanMessage(content="第二个呢")]}), "澄清")
        self.assertEqual(build.call_args.kwargs.get("extra_body"), {"enable_thinking": False})
        self.assertEqual(build.call_args.kwargs["max_tokens"], 256)
        self.assertEqual(build.call_args.kwargs["max_retries"], 0)
        output_format = build.call_args.kwargs["model_kwargs"]["response_format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["json_schema"]["strict"])
        schema = output_format["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"intent", "subjects", "needs_clarification"})

    async def test_length_finish_is_rejected_even_if_json_looks_complete(self):
        self.client.ainvoke.return_value = route_response("订单查询", [])
        self.client.ainvoke.return_value.response_metadata = {"finish_reason": "length"}
        with self.assertLogs("agent.router", level="WARNING") as logs:
            self.assertEqual(await self.router.router(self.state), "识别失败")
        self.assertIn("output_truncated", "".join(logs.output))
        self.client.ainvoke.assert_awaited_once()

    def setUp(self):
        self.client = AsyncMock()
        self.router = IntentRouter(self.client)
        self.state = {"messages": [HumanMessage(content="第二个有黑色吗")], "summary": "候选依次为商品A、商品B"}

    async def test_every_label_has_destination(self):
        self.assertEqual(set(ROUTE_MAP), set(ROUTE_NAMES) | {"识别失败"})
        self.assertEqual({name for name, node in ROUTE_MAP.items() if node == "route_reply_node"}, set(ROUTE_REPLIES))
        prompt = ROUTER_SYSTEM.format(intent_list=" / ".join(ROUTE_NAMES), history_status="没有")
        for label in ROUTE_NAMES:
            self.assertIn(label, prompt)
            self.client.ainvoke.return_value = route_response(label, ["A"] if label == "业务咨询" else ["A", "B"])
            self.assertEqual(await self.router.router(self.state), label)

    async def test_history_is_data_and_current_question_is_not_in_system(self):
        self.client.ainvoke.return_value = route_response("业务咨询", ["A"])
        state = {"messages": [HumanMessage(content="看耳机"), AIMessage(content="A和B"), HumanMessage(content="黑色款有货吗 UNIQUE_QUERY")],
                 "summary": "预算300"}
        result = await self.router.node(state)
        self.assertEqual(result, {"intent": "业务咨询", "final_reply": "", "route_subjects": ["A"], "category_action": {}})
        call = self.client.ainvoke.call_args
        messages = call.args[0]
        self.assertIn("预算300", messages[1].content)
        self.assertNotIn("UNIQUE_QUERY", messages[0].content)
        self.assertIn("UNIQUE_QUERY", messages[-1].content)
        self.assertEqual(call.kwargs["config"]["tags"], ["internal"])

    async def test_invalid_output_does_not_become_business_or_security_denial(self):
        for content in ("", "我认为是业务咨询", "订单查询\n业务咨询", "未知"):
            self.client.ainvoke.return_value = AIMessage(content=content)
            self.assertEqual(await self.router.router(self.state), "识别失败")
        self.assertEqual(route_with_fallback({"intent": "不存在"}), "识别失败")

    async def test_timeout_exception_and_cancel(self):
        self.client.ainvoke.side_effect = TimeoutError("private text")
        with self.assertLogs("agent.router", level="WARNING") as logs:
            self.assertEqual(await self.router.router(self.state), "识别失败")
        self.assertNotIn("private text", "".join(logs.output))

        async def wait(*args, **kwargs):
            await asyncio.Event().wait()

        self.client.ainvoke.side_effect = wait
        self.router.TIMEOUT = 0.01
        self.assertEqual(await self.router.router(self.state), "识别失败")
        self.client.ainvoke.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.router.router(self.state)

    async def test_empty_input_does_not_call_model(self):
        self.assertEqual(await self.router.router({"messages": []}), "澄清")
        self.client.ainvoke.assert_not_called()

    async def test_object_evidence_and_clarification_are_enforced(self):
        for intent, subjects, missing in [
            ("业务咨询", [], False), ("业务咨询", ["模型编的商品"], False),
            ("商品对比", ["A", "A"], False), ("商品对比", ["A"], False),
            ("业务咨询", ["A"], True),
            ("业务咨询", ["A", "B"], False),
        ]:
            with self.subTest(intent=intent, subjects=subjects):
                self.client.ainvoke.return_value = route_response(intent, subjects, missing)
                self.assertEqual(await self.router.router(self.state), "澄清")
        for intent in ("越权拒答", "转人工", "订单查询", "商品推荐", "概览筛选", "政策问答"):
            self.client.ainvoke.return_value = route_response(intent, [], True)
            self.assertEqual(await self.router.router(self.state), intent)
        self.client.ainvoke.return_value = route_response("业务咨询", ["蓝色版"])
        self.assertEqual(await self.router.router({"messages": [HumanMessage(content="上面那个蓝色版多少钱")]}), "澄清")

    async def test_invalid_decision_schema_is_not_executed(self):
        for data in [[], {"intent": "业务咨询"},
                     {"intent": "业务咨询", "subjects": "A", "needs_clarification": False},
                     {"intent": "业务咨询", "subjects": ["A"], "needs_clarification": "false"}]:
            self.client.ainvoke.return_value = AIMessage(content=json.dumps(data))
            self.assertEqual(await self.router.router(self.state), "识别失败")

    async def test_boundary_reply_graph_calls_router_once_and_never_business(self):
        for label in ROUTE_REPLIES:
            self.client.reset_mock()
            self.client.ainvoke.return_value = route_response(label)
            builder = StateGraph(AgentState)
            builder.add_node("intent", self.router.node)
            builder.add_node("route_reply_node", route_reply_node)
            business_calls = []

            async def business(state):
                business_calls.append(True)
                return {}

            for node in set(ROUTE_MAP.values()) - {"route_reply_node"}:
                builder.add_node(node, business)
                builder.add_edge(node, END)
            builder.add_edge(START, "intent")
            builder.add_conditional_edges("intent", route_with_fallback, ROUTE_MAP)
            builder.add_edge("route_reply_node", END)
            result = await builder.compile().ainvoke({**self.state, "intent": "商品推荐", "final_reply": "上一轮回复"})
            self.assertEqual(result["final_reply"], ROUTE_REPLIES[label])
            self.assertEqual(result["messages"][-1].content, result["final_reply"])
            self.assertEqual(self.client.ainvoke.await_count, 1)
            self.assertFalse(business_calls)

    async def test_production_graph_streams_boundary_reply(self):
        from tests.check_runtime_boundaries import app
        with app(":memory:") as (main, deps, products):
            client = AsyncMock()
            client.ainvoke.return_value = route_response("订单查询")
            with patch.object(main.router, "client", client):
                chunks = [chunk async for chunk in main.get_ai_stream("我的订单到哪了", "isolated-test")]
            self.assertEqual("".join(chunks), ROUTE_REPLIES["订单查询"])
            client.ainvoke.assert_awaited_once()
            deps.retriever.get_context.assert_not_called()
            products.resolve_product.assert_not_called()



if __name__ == "__main__":
    unittest.main()

class CandidateContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_state_counts_as_history_and_supports_full_name_evidence(self):
        client = AsyncMock()
        client.ainvoke.return_value = route_response('业务咨询', ['品牌甲完整型号Y'])
        result = await IntentRouter(client).decision({
            'messages': [HumanMessage(content='第二个多少钱')],
            'candidate_products': [{'product_id': 12, 'name': '品牌乙完整型号X'},
                                   {'product_id': 11, 'name': '品牌甲完整型号Y'}]})
        self.assertEqual(result, ('业务咨询', ['品牌甲完整型号Y']))
        sent = client.ainvoke.call_args.args[0]
        self.assertIn('历史：存在', sent[0].content)
        self.assertNotIn('当前消息是唯一业务信息', sent[0].content)
        self.assertIn('品牌甲完整型号Y', sent[1].content)
