"""离线验证：真实消息 reducer / SQLite checkpoint，模型使用固定替身。"""

import asyncio
from pathlib import Path
import sys
import os
import tempfile
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.memory import MemoryManager
from agent.state import AgentState


def conversation(turns):
    messages = []
    for n in range(turns):
        messages.extend([
            HumanMessage(content=f"用户需求{n}", id=f"h{n}"),
            AIMessage(content=f"回答{n}", id=f"a{n}"),
        ])
    return messages


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = AsyncMock()
        self.model.ainvoke.return_value = AIMessage(content="用户预算 300 元，偏好黑色耳机。")
        self.memory = MemoryManager(self.model)

    async def test_no_summary_call_within_window(self):
        self.assertEqual(await self.memory.summarize_node({"messages": conversation(4)}), {})
        self.model.ainvoke.assert_not_called()

    async def test_summary_removes_only_archived_messages(self):
        messages = conversation(6)
        update = await self.memory.summarize_node({"messages": messages, "summary": "喜欢黑色"})
        remaining = add_messages(messages, update["messages"])
        self.assertEqual([m.id for m in remaining], [m.id for m in messages[4:]])
        call = self.model.ainvoke.call_args
        self.assertIn("喜欢黑色", call.args[0][1].content)
        self.assertIn("用户需求0", call.args[0][1].content)
        self.assertNotIn("用户需求2", call.args[0][1].content)
        self.assertEqual(call.kwargs["config"]["tags"], ["internal"])
        self.assertEqual(await self.memory.summarize_node({"messages": remaining, "summary": update["summary"]}), {})
        self.assertEqual(self.model.ainvoke.await_count, 1)

    async def test_failed_summary_preserves_original(self):
        self.model.ainvoke.side_effect = TimeoutError("sensitive payload must not be logged")
        messages = conversation(6)
        with self.assertLogs("agent.memory", level="WARNING") as logs:
            update = await self.memory.summarize_node({"messages": messages, "summary": "旧摘要"})
        self.assertEqual(update, {})
        self.assertEqual(len(messages), 12)
        self.assertNotIn("sensitive payload", "".join(logs.output))
        self.assertLessEqual(len(self.memory.build_messages({"messages": messages}, "规则")), 13)

    async def test_invalid_summary_never_deletes(self):
        for content in ("", " " * 5, "长" * 1201):
            with self.subTest(content_length=len(content)):
                self.model.ainvoke.return_value = AIMessage(content=content)
                self.assertEqual(await self.memory.summarize_node({"messages": conversation(6)}), {})

    async def test_timeout_and_cancellation(self):
        cancelled = asyncio.Event()

        async def hanging(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.model.ainvoke.side_effect = hanging
        self.memory.SUMMARY_TIMEOUT = 0.01
        self.assertEqual(await self.memory.summarize_node({"messages": conversation(6)}), {})
        self.assertTrue(cancelled.is_set())
        self.model.ainvoke.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.memory.summarize_node({"messages": conversation(6)})

    async def test_short_term_keeps_latest_question_once(self):
        messages = conversation(8) + [HumanMessage(content="第二个呢", id="latest")]
        result = self.memory.build_messages({"messages": messages, "summary": "预算300"}, "规则")
        self.assertIsInstance(result[0], SystemMessage)
        self.assertIsInstance(result[1], HumanMessage)
        self.assertIn("预算300", result[1].content)
        self.assertEqual(sum(m.content == "第二个呢" for m in result), 1)
        self.assertEqual(result[-1].id, "latest")
        self.assertIsInstance(result[2], HumanMessage)
        self.assertLessEqual(len(result[2:]), 12)

    async def test_character_budget_keeps_whole_recent_turns(self):
        messages = [HumanMessage(content="旧" * 5000, id="old"), AIMessage(content="旧答", id="oa"),
                    HumanMessage(content="新" * 2000, id="new"), AIMessage(content="新答", id="na")]
        recent = self.memory.recent_messages(messages)
        self.assertEqual([m.id for m in recent], ["new", "na"])
        self.assertEqual(recent[0].content, "新" * 2000)

    async def test_summary_batch_is_bounded_and_leaves_unprocessed_history(self):
        self.memory.MAX_SUMMARY_INPUT_CHARS = 12
        messages = conversation(10)
        update = await self.memory.summarize_node({"messages": messages})
        removed = {m.id for m in update["messages"]}
        self.assertTrue(removed)
        self.assertLess(len(removed), len(messages) - 8)
        self.assertLessEqual(sum(len(m.content) for m in messages if m.id in removed), 12)

    async def test_missing_ids_do_not_delete(self):
        messages = conversation(6)
        messages[0].id = None
        self.assertEqual(await self.memory.summarize_node({"messages": messages}), {})
        self.model.ainvoke.assert_not_called()

    async def test_empty_backlog_has_message_count_limit(self):
        messages = conversation(100)
        for message in messages:
            message.content = ""
        update = await self.memory.summarize_node({"messages": messages})
        self.assertEqual(len(update["messages"]), 32)

    async def test_failure_can_recover_without_losing_previous_summary(self):
        state = {"messages": conversation(6), "summary": "用户拒绝推荐"}
        self.model.ainvoke.side_effect = TimeoutError("offline")
        self.assertEqual(await self.memory.summarize_node(state), {})
        self.model.ainvoke.side_effect = None
        update = await self.memory.summarize_node(state)
        self.assertIn("用户拒绝推荐", self.model.ainvoke.call_args.args[0][1].content)
        self.assertEqual(len(add_messages(state["messages"], update["messages"])), 8)

    async def test_all_message_builders_share_summary_and_window(self):
        state = {"messages": conversation(8) + [HumanMessage(content="当前问题")], "summary": "偏好黑色"}
        outputs = [
            self.memory.build_messages(state, "咨询规则"),
            self.memory.build_messages(state, "闲聊规则"),
            self.memory.build_messages(state, "推荐规则"),
        ]
        for messages in outputs:
            self.assertTrue(any("偏好黑色" in message.content for message in messages[1:]))
            self.assertEqual(messages[-1].content, "当前问题")
            self.assertLessEqual(len(messages), 15)

    async def test_sqlite_restart_and_thread_isolation(self):
        async def answer(state):
            return {"messages": [AIMessage(content="收到") ],
                    "task_context": {"intent": "概览筛选", "filters": {"max_price": 300}}}

        builder = StateGraph(AgentState)
        builder.add_node("answer", answer)
        builder.add_node("summary", self.memory.summarize_node)
        builder.add_edge(START, "answer")
        builder.add_edge("answer", "summary")
        builder.add_edge("summary", END)
        config_a = {"configurable": {"thread_id": "A"}}
        config_b = {"configurable": {"thread_id": "B"}}
        fd, path = tempfile.mkstemp(suffix=".db", dir=Path(__file__).parent)
        os.close(fd)
        try:
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = builder.compile(checkpointer=saver)
                for n in range(6):
                    await graph.ainvoke({"messages": [HumanMessage(content=f"需求{n}")]}, config_a)
                saved = (await graph.aget_state(config_a)).values
                self.assertEqual(len(saved["messages"]), 8)
                self.assertIn("预算", saved["summary"])
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = builder.compile(checkpointer=saver)
                restored = (await graph.aget_state(config_a)).values
                self.assertEqual(restored["summary"], saved["summary"])
                self.assertEqual(restored["task_context"], saved["task_context"])
                self.assertNotIn("task_context", (await graph.aget_state(config_b)).values)
                self.assertEqual([m.id for m in restored["messages"]], [m.id for m in saved["messages"]])
                result_b = await graph.ainvoke({"messages": [HumanMessage(content="我是新会话")]}, config_b)
                self.assertFalse(result_b.get("summary"))
                self.assertEqual(len(result_b["messages"]), 2)
                self.assertEqual((await graph.aget_state(config_a)).values["summary"], saved["summary"])
        finally:
            Path(path).unlink(missing_ok=True)



if __name__ == "__main__":
    unittest.main()

class PurposeContextTests(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryManager()

    def test_extract_keeps_cancelled_state_and_last_question_without_old_summary(self):
        state = {'messages': conversation(5) + [HumanMessage(content='预算提高到500元')],
                 'summary': '旧预算100，必须恢复黑色', 'intent': '商品推荐',
                 'task_context': {'intent': '概览筛选', 'filters': {
                     'category': '耳机', 'max_price': None, 'color': None, 'unsupported_specs': []},
                     'color_choices': ['曜石黑', '白色'], 'product_names': ['无关旧商品']}}
        result = self.memory.build_messages(state, '提取规则', purpose='extract')
        text = '\n'.join(m.content for m in result)
        self.assertNotIn('旧预算100', text)
        self.assertNotIn('无关旧商品', text)
        self.assertIn('"color":null', text)
        self.assertIn('曜石黑', text)
        self.assertEqual([m.content for m in result[-3:]], ['用户需求4', '回答4', '预算提高到500元'])
        self.assertEqual(result[0].content, '提取规则')

    def test_route_keeps_candidate_order_and_selected_object_outside_history_window(self):
        state = {'messages': conversation(8) + [HumanMessage(content='第二个有货吗')],
                 'task_context': {'intent': '业务咨询', 'product_names': ['完整商品乙型号X'],
                                  'color_choices': ['与路由无关的颜色选项']},
                 'candidate_products': [{'product_id': 12, 'name': '完整商品乙型号X'},
                                        {'product_id': 11, 'name': '完整商品甲型号Y'}]}
        result = self.memory.build_messages(state, '路由规则', purpose='route')
        context = result[1].content
        self.assertLess(context.index('"product_id":12'), context.index('"product_id":11'))
        self.assertIn('"product_names":["完整商品乙型号X"]', context)
        self.assertNotIn('与路由无关', context)
        self.assertEqual(result[-1].content, '第二个有货吗')
        self.assertEqual(result[0].content, '路由规则')

    def test_extract_without_filters_keeps_summary_for_unresolved_reference(self):
        state = {'summary': '此前想买黑色耳机', 'messages': [HumanMessage(content='推荐刚才那类')],
                 'task_context': {'intent': '闲聊'}}
        result = self.memory.build_messages(state, '提取规则', purpose='extract')
        self.assertTrue(any('此前想买黑色耳机' in m.content for m in result[1:]))
        self.assertEqual(result[-1].content, '推荐刚才那类')

    def test_answer_retains_history_and_fact_boundaries(self):
        state = {'summary': '喜欢轻便', 'messages': conversation(5) + [HumanMessage(content='它多重')],
                 'intent': '业务咨询', 'task_context': {'intent': '业务咨询', 'product_names': ['耳机甲']}}
        result = self.memory.build_messages(state, '回答规则')
        self.assertIn('价格、库存、订单必须重新查询', result[0].content)
        self.assertTrue(any('喜欢轻便' in m.content for m in result[1:]))
        self.assertTrue(any('用户需求0' in m.content for m in result[1:]))
        self.assertEqual(result[-1].content, '它多重')

    def test_compact_context_keeps_long_current_message_intact(self):
        query = '长问题' * 1500
        for purpose in ('route', 'extract'):
            with self.subTest(purpose=purpose):
                result = self.memory.build_messages({'messages': [HumanMessage(content=query)],
                    'task_context': {'intent': '商品推荐', 'filters': {'color': None}}}, '规则', purpose=purpose)
                self.assertEqual(result[-1].content, query)
