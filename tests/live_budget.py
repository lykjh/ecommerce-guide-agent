"""预算入口真实模型冒烟：仅一次真实路由，其余验证发送前拦截。"""
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.budget import TurnBudget, BUDGET_REPLY
from agent.router import APP_CFG
from agent.prompts import ROUTE_REPLIES
from langchain.chat_models import init_chat_model
from check_runtime_boundaries import app, collect


async def main():
    destination = Path(__file__).resolve().parents[1] / "docs" / "06-问题发现与解决" / "2026-11-01" / "证据"
    handler = logging.FileHandler(destination / "budget-live.log", mode="w", encoding="utf-8")
    loggers = [logging.getLogger(name) for name in ("agent.budget", "agent.telemetry")]
    for logger in loggers:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    results = []
    try:
        client = init_chat_model(model=APP_CFG.get("llm.router_model"), model_provider=APP_CFG.get("llm.model_provider"),
            api_key=os.getenv(APP_CFG.get("llm.api_key_env")), base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
            temperature=0, max_tokens=256, max_retries=0, extra_body=APP_CFG.get("llm.router_extra_body", {}))
        with app(":memory:") as (main, _, _), patch.object(main.router, "client", client):
            reply = await collect(main, "我的订单到哪了", "budget-live-synthetic")
            results.append({"id": "real_model_via_entry", "passed": reply == ROUTE_REPLIES["订单查询"]})
            with patch.object(TurnBudget, "MAX_CALLS", 0):
                reply = await collect(main, "我的订单到哪了", "budget-block-before-send")
                results.append({"id": "zero_calls_blocks_before_send", "passed": BUDGET_REPLY in reply})
            results.append({"id": "session_released", "passed": not main._active_sessions})
        handler.flush()
        text = (destination / "budget-live.log").read_text(encoding="utf-8")
        results.append({"id": "usage_settled_and_blocked_count_zero", "passed":
                        "estimated=False" in text and "calls=0 queries=0 charged_tokens=0" in text
                        and "calls=1 input_tokens=" in text and "llm_usage" in text})
    except Exception as exc:
        results.append({"id": "service_error", "passed": False, "error_type": type(exc).__name__})
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
        handler.close()
    report = {"model": APP_CFG.get("llm.router_model"), "cases": results, "passed": all(r["passed"] for r in results)}
    (destination / "budget-live-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report["passed"]


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
