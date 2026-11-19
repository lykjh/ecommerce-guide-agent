"""H独占的模型/商城替身；保留真实HTTP、图、预算、记忆和SQLite。"""
import asyncio
from contextlib import ExitStack, contextmanager
import json
import logging
import re
from typing import Any
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import main_agent
from agent.budget import CURRENT_BUDGET
from agent.utils.log_tool import get_session_id
from tests.mall_fixtures import detail

MARKERS = tuple(f"SESSION_{chr(65 + i)}" for i in range(10))


class OfflineModel(BaseChatModel):
    mode: str
    control: Any

    @property
    def _llm_type(self):
        return "h-offline-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("测试模型只允许异步执行")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        text = "\n".join(message.content for message in messages)
        markers = set(re.findall(r"SESSION_[A-J]", text))
        assert len(markers) == 1, f"模型上下文串话: {sorted(markers)}"
        marker = markers.pop()
        budget = CURRENT_BUDGET.get()
        assert budget is not None
        self.control["calls"].append((self.mode, marker, get_session_id(), budget.request_id))
        if self.mode == "route":
            if self.control.get("barrier"):
                await asyncio.wait_for(self.control["barrier"].wait(), 10)
            content = json.dumps({"intent": "业务咨询", "subjects": ["商品" + marker], "needs_clarification": False}, ensure_ascii=False)
        elif self.mode == "summary":
            self.control["summaries"].append(marker)
            content = marker + "的历史摘要：只保留本人商品需求。"
        else:
            latest = next(message.content for message in reversed(messages) if isinstance(message, HumanMessage))
            latest = latest.removeprefix("仅回答本轮问题，不补答历史未完成或已取消的任务：\n")
            if "CANCEL" in latest:
                self.control["entered"].set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.control["closed"].set()
            content = f"{marker}独立答复：{latest}"
        message = AIMessage(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return ChatResult(generations=[ChatGeneration(message=message)])


@contextmanager
def offline_runtime(path, control=None):
    control = control if control is not None else {"calls": [], "summaries": []}
    with ExitStack() as stack:
        stack.enter_context(patch.object(main_agent, "DB_PATH", str(path)))
        stack.enter_context(patch.object(main_agent.router, "client", OfflineModel(mode="route", control=control)))
        stack.enter_context(patch.object(main_agent.memory_manager, "summarizer_client", OfflineModel(mode="summary", control=control)))
        stack.enter_context(patch("agent.nodes.business.get_client", return_value=OfflineModel(mode="answer", control=control)))
        stack.enter_context(patch("agent.tool.mall.get_product", side_effect=lambda product_id: detail(product_id, "商品" + MARKERS[product_id - 1001])))
        stack.enter_context(patch("agent.tool.mall._get", side_effect=AssertionError("禁止在线商城访问")))
        yield control


async def seed_sessions(path, sessions):
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await saver.setup()
        graph = main_agent.builder.compile(checkpointer=saver)
        for index, session_id in enumerate(sessions):
            marker = MARKERS[index]
            history = []
            for turn in range(4):
                history.extend([HumanMessage(content=f"{marker}历史需求{turn}"), AIMessage(content=f"{marker}历史答复{turn}")])
            await graph.aupdate_state({"configurable": {"thread_id": session_id}}, {
                "messages": history, "summary": f"{marker}初始摘要", "intent": "业务咨询",
                "route_subjects": ["商品" + marker], "final_reply": marker + "旧回复",
                "task_context": {"intent": "业务咨询", "product_names": ["商品" + marker]},
                "candidate_products": [{"product_id": 1001 + index, "name": "商品" + marker}],
                "product_page": {"filters": {"category": marker, "max_price": str(300 + index)}, "hasMore": True, "nextCursor": marker + "游标"},
                "signal_state": {"signal": "品类偏好", "level": index % 7, "source": marker},
            }, as_node="summarize_node")


async def snapshot(path, session_id):
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        state = await main_agent.builder.compile(checkpointer=saver).aget_state({"configurable": {"thread_id": session_id}})
        return state.values


def decode_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(":", 1) for line in block.splitlines())
        events.append((lines["event"].strip(), json.loads(lines["data"])))
    return events


class TraceCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.entries = []

    def emit(self, record):
        self.entries.append((record.name, get_session_id(), record.getMessage()))
