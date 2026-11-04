"""显式在线复现：相同路由输入，仅切换 enable_thinking，保留用量与结束原因。"""
import asyncio
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.memory import APP_CFG
from agent.state import AgentState
from agent.router import IntentRouter
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

CASES = [
    ("ambiguous", "第二个有黑色吗", "", "澄清"),
    ("switch_topic", "我的订单到哪了", "用户之前正在挑选黑色耳机。", "订单查询"),
    ("multiple", "查我的订单，再比较耳机A和耳机B", "", "订单查询"),
]


class Capture(BaseCallbackHandler):
    def __init__(self):
        self.result = {}

    def on_llm_end(self, response, **kwargs):
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        self.result = {"finish_reason": message.response_metadata.get("finish_reason"),
                       "content": message.content, "usage": usage}


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    rows = []
    for repeat in range(3):
        for name, query, summary, expected in CASES:
            options = {"extra_body": {"enable_thinking": False}} if mode == "no-thinking" else {}
            client = init_chat_model(model=APP_CFG.get("llm.router_model"), model_provider=APP_CFG.get("llm.model_provider"),
                api_key=os.getenv(APP_CFG.get("llm.api_key_env")), base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
                temperature=0, max_tokens=256, max_retries=0, **options)
            router = IntentRouter(None if mode == "fixed" else client)
            capture = Capture()
            builder = StateGraph(AgentState)
            builder.add_node("route", router.node)
            builder.add_edge(START, "route")
            builder.add_edge("route", END)
            result = await builder.compile().ainvoke({"messages": [HumanMessage(content=query)], "summary": summary},
                config={"callbacks": [capture]})
            row = {"mode": mode, "repeat": repeat, "case": name, "expected": expected,
                   "actual": result["intent"], "passed": result["intent"] == expected, **capture.result}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (Path(__file__).resolve().parents[1] / "docs" / "09-验收与交付" / "证据" / "专项测试" / f"router-truncation-{mode}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            if not capture.result:
                return False
    return all(row["passed"] for row in rows)


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
