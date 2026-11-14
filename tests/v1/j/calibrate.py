"""离线 ASGI 与真实图/临时 SQLite 校准；没有在线地址参数。"""
import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack, aclosing, closing
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
import time
import tracemalloc
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from agent import main_agent
from agent.budget import BUDGET_REPLY
from agent.tool.schemas import FilterParams
from tests.check_runtime_boundaries import offline_http_client
from tests.mall_fixtures import detail, selection

OFFLINE_REPLY = "OFFLINE_TRANSPORT_OK"


def classify(status, payload, expected):
    """按实际终止事件和回复判定，HTTP 200/done 不自动算成功。"""
    if status == 429:
        return "rate_limited"
    if status in (408, 504):
        return "timeout"
    if status != 200:
        return "http_failure"
    frames = []
    for frame in payload.strip().split("\n\n"):
        if not frame:
            continue
        fields = dict(line.split(":", 1) for line in frame.splitlines())
        frames.append((fields["event"].strip(), json.loads(fields["data"])))
    names = [name for name, _ in frames]
    text = "".join(data["text"] for name, data in frames if name == "delta")
    if "error" in names:
        return "stream_failure"
    if not names or names[-1] != "done":
        return "incomplete"
    if main_agent.SESSION_BUSY_REPLY in text:
        return "session_busy"
    if BUDGET_REPLY in text:
        return "budget_stopped"
    return "fixture_success" if expected in text else "unexpected_reply"


def summarize(samples, wall_seconds):
    """Nearest-rank分位数；吞吐包含全部响应，成功吞吐另算。"""
    durations = sorted(item["elapsed_ms"] for item in samples)
    outcomes = Counter(item["outcome"] for item in samples)
    return {
        "requests": len(samples), "outcomes": dict(outcomes),
        "p50_ms": durations[math.ceil(len(durations) * .50) - 1],
        "p95_ms": durations[math.ceil(len(durations) * .95) - 1],
        "requests_per_second": len(samples) / wall_seconds,
        "fixture_success_per_second": outcomes["fixture_success"] / wall_seconds,
        "wall_seconds": wall_seconds,
    }


async def calibrate(mode, concurrency, rounds):
    samples = []
    current = peak = server_current = server_peak = 0
    original = main_agent.stream_reply
    transport_barrier = asyncio.Barrier(concurrency)

    async def measured_stream(*args, **kwargs):
        nonlocal server_current, server_peak
        server_current += 1
        server_peak = max(server_peak, server_current)
        try:
            if mode == "transport":
                await transport_barrier.wait()
                yield OFFLINE_REPLY
            else:
                async with aclosing(original(*args, **kwargs)) as stream:
                    async for part in stream:
                        yield part
        finally:
            server_current -= 1

    async def filters(messages, config):
        return FilterParams(category="耳机", max_price=500)

    with tempfile.TemporaryDirectory(prefix="agent-offline-capacity-") as directory, ExitStack() as stack:
        db = Path(directory) / "checkpoint.db"
        stack.enter_context(patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}))
        stack.enter_context(patch.object(main_agent, "DB_PATH", str(db)))
        stack.enter_context(patch("agent.http_api.stream_reply", measured_stream))
        stack.enter_context(patch("socket.socket.connect", side_effect=AssertionError("offline calibration forbids network")))
        router = FakeListChatModel(responses=[json.dumps({"intent": "商品推荐", "subjects": [], "needs_clarification": False}, ensure_ascii=False)])
        stack.enter_context(patch.object(main_agent.router, "client", router))
        extract = MagicMock()
        extract.with_structured_output.return_value.with_config.return_value = RunnableLambda(filters)
        stack.enter_context(patch("agent.nodes.recommend.get_extract_client", return_value=extract))
        stack.enter_context(patch("agent.tool.mall.list_categories", new=AsyncMock(return_value=[])))
        stack.enter_context(patch("agent.nodes.recommend.find_products", new=AsyncMock(return_value=selection(detail()))))
        stack.enter_context(patch.object(main_agent.memory_manager, "_summary_client", return_value=FakeListChatModel(responses=["离线测试耳机预算500元"])) )
        async with offline_http_client(db) as client:
            tokens = [(await client.post("/api/sessions")).raise_for_status().json()["session_token"] for _ in range(concurrency)]
            expected = OFFLINE_REPLY if mode == "transport" else "商品编号 101"
            tracemalloc.start()
            cpu_start, start = time.process_time(), time.perf_counter()
            for round_index in range(rounds):
                barrier = asyncio.Barrier(concurrency)

                async def request(index):
                    nonlocal current, peak
                    await barrier.wait()
                    current += 1
                    peak = max(peak, current)
                    began = time.perf_counter()
                    try:
                        # ASGITransport不使用套接字；显式asyncio期限避免误以为httpx timeout约束了ASGI。
                        async with asyncio.timeout(35):
                            response = await client.post("/api/chat", headers={"Authorization": "Bearer " + tokens[index]}, json={"message": "推荐500元以内的耳机"})
                        outcome = classify(response.status_code, response.text, expected)
                    except TimeoutError:
                        outcome = "timeout"
                    finally:
                        current -= 1
                    samples.append({"round": round_index + 1, "client": index, "outcome": outcome, "elapsed_ms": (time.perf_counter() - began) * 1000})

                await asyncio.gather(*(request(index) for index in range(concurrency)))
            elapsed = time.perf_counter() - start
            cpu_seconds = time.process_time() - cpu_start
            _, heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        db_bytes = sum(path.stat().st_size for path in Path(directory).glob("checkpoint.db*"))
        checkpoint_rows = 0
        if mode == "graph":
            with closing(sqlite3.connect(db)) as connection:
                checkpoint_rows = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        return {
            "mode": mode, "concurrency_target": concurrency, "rounds": rounds,
            "actual_client_peak_in_flight": peak, "actual_server_peak_streams": server_peak,
            **summarize(samples, elapsed), "cpu_process_seconds": cpu_seconds,
            "python_heap_peak_bytes": heap_peak, "sqlite_file_bytes_after": db_bytes,
            "checkpoint_rows": checkpoint_rows, "samples": samples,
            "not_measured": ["socket/proxy buffering", "RSS and system CPU", "DB lock wait and physical IO", "real model/mall latency and cost", "production capacity"],
            "budget_stop_reason_limit": "当前SSE预算回复未区分timeout/tokens/query_busy，统一计budget_stopped；不能猜具体原因。",
        }


async def run(rounds):
    return {
        "scope": "offline_in_process_ASGI_and_real_graph_temp_sqlite",
        "python": platform.python_version(), "platform": platform.platform(),
        "network": "forbidden", "real_model_calls": 0, "real_mall_calls": 0,
        "instrumentation": "tracemalloc and normal logging enabled; startup inside first round; no warmup; diagnostic latency only",
        "fixtures": "router FakeListChatModel; extractor RunnableLambda; product AsyncMock; summary FakeListChatModel",
        "results": [await calibrate(mode, size, rounds) for mode in ("transport", "graph") for size in (1, 5, 10)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=3)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.rounds)), ensure_ascii=False, indent=2))
