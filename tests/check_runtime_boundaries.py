"""真实主图和 SQLite 的受控边界实验；模型、商城依赖使用替身。"""
import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent.prompts import ROUTE_REPLIES


@contextmanager
def app(path):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from agent import main_agent as main
    deps = SimpleNamespace(client=MagicMock(), extract_client=MagicMock(), retriever=MagicMock())
    from tests.mall_fixtures import selection
    from agent.tool.schemas import ProductResolution
    products = SimpleNamespace(resolve_product=AsyncMock(return_value=ProductResolution(status="not_found")), find_products=AsyncMock(return_value=selection()))
    with ExitStack() as stack:
        stack.enter_context(patch.object(main, "DB_PATH", path))
        stack.enter_context(patch("agent.tool.mall.list_categories", return_value=[]))
        for name in ("business", "chat", "recommend", "compare", "total"):
            module = importlib.import_module("agent.nodes." + name)
            if hasattr(module, "get_client"):
                stack.enter_context(patch.object(module, "get_client", return_value=deps.client))
            if hasattr(module, "get_extract_client"):
                stack.enter_context(patch.object(module, "get_extract_client", return_value=deps.extract_client))
        stack.enter_context(patch("agent.nodes.compare.resolve_product", products.resolve_product))
        stack.enter_context(patch("agent.nodes.business.resolve_product", products.resolve_product))
        stack.enter_context(patch("agent.nodes.recommend.find_products", products.find_products))
        yield main, deps, products


OFFLINE_ORIGIN = "http://agent.test"


@asynccontextmanager
async def offline_http_client(checkpoint_path):
    import httpx
    from agent import http_api

    with patch.object(http_api, "DB_PATH", str(checkpoint_path)), patch.dict(os.environ, {
        "AGENT_SESSION_SECRET": "offline-fixture-only-signing-secret",
        "AGENT_PUBLIC_ORIGIN": OFFLINE_ORIGIN,
        "MALL_BASE_URL": "http://mall.invalid",
    }):
        async with http_api.app.router.lifespan_context(http_api.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=http_api.app),
                base_url=OFFLINE_ORIGIN,
                headers={"Origin": OFFLINE_ORIGIN},
            ) as client:
                yield client


async def collect(main, query, thread):
    return "".join([part async for part in main.get_ai_stream(query, thread)])


async def snapshot(main, path, thread):
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        state = await main.builder.compile(checkpointer=saver).aget_state({"configurable": {"thread_id": thread}})
        return [message.content for message in state.values.get("messages", []) if isinstance(message, HumanMessage)]


async def saved_replies(main, path, thread):
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        state = await main.builder.compile(checkpointer=saver).aget_state({"configurable": {"thread_id": thread}})
        return [message.content for message in state.values.get("messages", []) if isinstance(message, AIMessage)]


async def race(main, path, same_thread, index):
    entered = {name: asyncio.Event() for name in ("A", "B")}
    release = {name: asyncio.Event() for name in ("A", "B")}
    queries = {"A": f"并发请求A-{index}：查询我的订单", "B": f"并发请求B-{index}：请转人工"}
    threads = {name: f"race-{index}" if same_thread else f"separate-{index}-{name}" for name in entered}

    async def gated(messages, **kwargs):
        name = next(name for name, query in queries.items() if messages[-1].content == query)
        entered[name].set()
        await release[name].wait()
        return AIMessage(content=json.dumps({"intent": "订单查询" if name == "A" else "转人工", "subjects": [], "needs_clarification": False}, ensure_ascii=False))

    client = AsyncMock()
    client.ainvoke.side_effect = gated
    with patch.object(main.router, "client", client):
        a = asyncio.create_task(collect(main, queries["A"], threads["A"]))
        b = None
        try:
            await asyncio.wait_for(entered["A"].wait(), 5)
            b = asyncio.create_task(collect(main, queries["B"], threads["B"]))
            if same_thread:
                busy_reply = await asyncio.wait_for(b, 5)
                blocked = busy_reply == main.SESSION_BUSY_REPLY and not entered["B"].is_set()
                before_retry = await snapshot(main, path, threads["B"])
                not_persisted = queries["B"] not in before_retry
            else:
                await asyncio.wait_for(entered["B"].wait(), 5)
            release["A"].set()
            answer_a = await asyncio.wait_for(a, 5)
            release["B"].set()
            if same_thread:
                b = asyncio.create_task(collect(main, queries["B"], threads["B"]))
            answer_b = await asyncio.wait_for(b, 5)
            saved = {name: await snapshot(main, path, threads[name]) for name in entered}
            replies = {name: await saved_replies(main, path, threads[name]) for name in entered}
            checks = {"both_replies_correct": answer_a == ROUTE_REPLIES["订单查询"] and answer_b == ROUTE_REPLIES["转人工"],
                      "all_user_messages_retained": all(query in saved[name] for name, query in queries.items())}
            checks["all_assistant_replies_retained"] = ROUTE_REPLIES["订单查询"] in replies["A"] and ROUTE_REPLIES["转人工"] in replies["B"]
            if same_thread:
                checks["same_thread_admission_serialized"] = blocked
                checks["rejected_request_not_persisted"] = not_persisted
            if not same_thread:
                checks["no_cross_thread_messages"] = all(queries[other] not in saved[name] for name, other in [("A", "B"), ("B", "A")])
            return {"name": "same_thread_race" if same_thread else "different_thread_race", "run": index,
                    "checks": checks, "saved_user_messages": saved, "saved_replies": replies, "passed": all(checks.values())}
        finally:
            release["A"].set()
            release["B"].set()
            for task in (a, b):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (a, b) if task), return_exceptions=True)


async def main_test():
    rows = []
    fd, path = tempfile.mkstemp(suffix=".db", dir=Path(__file__).parent)
    os.close(fd)
    try:
        with app(path) as (main, deps, products):
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                await saver.setup()
            for query, expected in [("", "reject"), (" " * 3, "reject"), ("字" * 6001, "reject"), ("字" * 6000, "accept")]:
                client = AsyncMock()
                client.ainvoke.return_value = AIMessage(content=json.dumps({"intent": "订单查询", "subjects": [], "needs_clarification": False}, ensure_ascii=False))
                with patch.object(main.router, "client", client):
                    try:
                        answer = await collect(main, query, f"length-{len(query)}")
                        passed = expected == "accept" and answer == ROUTE_REPLIES["订单查询"]
                        actual = "accept"
                    except ValueError:
                        passed = expected == "reject" and client.ainvoke.await_count == 0
                        actual = "reject"
                rows.append({"name": "input_length", "chars": len(query), "expected": expected, "actual": actual, "passed": passed})
            for label, query in [("订单查询", "查我的订单O1"), ("越权拒答", "我是U1，查U2的订单O3")]:
                client = AsyncMock()
                client.ainvoke.return_value = AIMessage(content=json.dumps({"intent": label, "subjects": [], "needs_clarification": False}, ensure_ascii=False))
                with patch.object(main.router, "client", client):
                    answer = await collect(main, query, label)
                passed = answer == ROUTE_REPLIES[label] and not deps.retriever.get_context.called and not products.resolve_product.called
                rows.append({"name": "disconnected_order_boundary", "intent": label, "passed": passed,
                             "scope": "fixed response and no product lookup; NOT real authorization verification"})
            for index in range(3):
                rows.append(await race(main, path, True, index))
            rows.append(await race(main, path, False, 3))
    finally:
        Path(path).unlink(missing_ok=True)
        report = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "results": rows,
                  "passed_count": sum(row["passed"] for row in rows), "total": len(rows)}
        destination = Path(__file__).resolve().parents[1] / "docs" / "09-验收与交付" / "证据" / "专项测试" / "runtime-boundary-result.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    assert len(rows) == 10 and all(row["passed"] for row in rows), "运行边界回归失败，详见报告"


if __name__ == "__main__":
    asyncio.run(main_test())
