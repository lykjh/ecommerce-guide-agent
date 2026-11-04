"""真实路由及回答模型，固定商品检索fixture；不触碰生产会话库。"""
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.budget import TurnBudget, budget_scope
from agent.state import AgentState
from agent.router import APP_CFG, IntentRouter
from agent.nodes.routing import route_reply_node
from agent.retrieval import NO_EVIDENCE, AMBIGUOUS_PRODUCT
from agent.telemetry import TurnUsage
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from check_runtime_boundaries import app
from test_rag_multiturn import A, B

DEST = Path(__file__).resolve().parents[1] / "docs/06-问题发现与解决/2026-11-01/证据"


class Capture(BaseCallbackHandler):
    def __init__(self):
        self.outputs = []

    def on_llm_end(self, response, **kwargs):
        self.outputs.append(response.generations[0][0].message.content)


async def main():
    client = init_chat_model(model=APP_CFG.get("llm.model"), model_provider=APP_CFG.get("llm.model_provider"),
        api_key=os.getenv(APP_CFG.get("llm.api_key_env")), base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
        temperature=0, max_tokens=500, max_retries=0)
    loggers = [logging.getLogger(name) for name in ("agent.budget", "agent.telemetry", "agent.nodes.business")]
    handler = logging.FileHandler(DEST / "rag-live.log", mode="w", encoding="utf-8")
    for logger in loggers:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    fd, path = tempfile.mkstemp(suffix=".db", dir=Path(__file__).parent)
    os.close(fd)
    rows = []
    try:
        with app(path) as (_, deps, _):
            from agent.nodes.business import business_node
            deps.client.ainvoke = client.ainvoke
            router = IntentRouter()
            builder = StateGraph(AgentState)
            builder.add_node("route", router.node)
            builder.add_node("business", business_node)
            builder.add_node("fixed", route_reply_node)
            builder.add_edge(START, "route")
            builder.add_conditional_edges("route", lambda s: "business" if s["intent"] == "业务咨询" else "fixed")
            builder.add_edge("business", END)
            builder.add_edge("fixed", END)
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = builder.compile(checkpointer=saver)
                cases = [
                    ("second", "same", "第二个续航怎样", "业务咨询", "测试耳机乙", "12", A + "\n\n" + B),
                    ("pronoun", "same", "它有白色吗", "业务咨询", "测试耳机乙", "黑色", A + "\n\n" + B),
                    ("change_subject", "same", "改问测试耳机甲，续航多久", "业务咨询", "测试耳机甲", "8", A + "\n\n" + B),
                    ("switch_order", "same", "查我的订单", "订单查询", None, None, ""),
                    ("new_session", "new", "它有白色吗", "澄清", None, None, ""),
                    ("wrong_evidence", "wrong", "测试耳机甲续航多久", "业务咨询", "测试耳机甲", NO_EVIDENCE, B),
                    ("duplicate_name", "duplicate", "测试耳机甲续航多久", "业务咨询", "测试耳机甲", AMBIGUOUS_PRODUCT, A + "\n\n" + A.replace("G-9001", "G-9003")),
                ]
                for name, thread, query, intent, subject, expected, context in cases:
                    deps.retriever.get_context.reset_mock()
                    deps.retriever.get_context.return_value = context
                    messages = [HumanMessage(content=query)]
                    if name == "second":
                        messages = [HumanMessage(content="对比测试耳机甲和测试耳机乙"), AIMessage(content="候选依次是测试耳机甲、测试耳机乙。"), *messages]
                    usage = TurnUsage(name)
                    capture = Capture()
                    budget = TurnBudget(name)
                    try:
                        with budget_scope(budget):
                            async with asyncio.timeout_at(budget.deadline):
                                result = await graph.ainvoke({"messages": messages}, config={"configurable": {"thread_id": thread}, "callbacks": [usage, capture]})
                        actual_query = deps.retriever.get_context.call_args.args[0] if deps.retriever.get_context.called else ""
                        answer = result.get("final_reply", "")
                        checks = {"intent": result["intent"] == intent,
                                  "subject": result.get("route_subjects", []) == ([subject] if subject else []),
                                  "retrieval": (subject in actual_query) if subject else not deps.retriever.get_context.called,
                                  "answer": expected in answer if expected else True,
                                  "calls": usage.calls == (2 if name in ("second", "pronoun", "change_subject") else 1)}
                        if name == "switch_order":
                            checks["cleared"] = not result.get("task_context")
                        if name == "second":
                            checks["no_wrong_fact"] = "8小时" not in answer
                        row = {"id": name, "checks": checks, "passed": all(checks.values()), "answer": answer,
                               "retrieval_query": actual_query, "route_output": capture.outputs[0] if capture.outputs else None,
                               "calls": usage.calls, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
                    except Exception as exc:
                        row = {"id": name, "passed": False, "error_type": type(exc).__name__}
                    finally:
                        usage.finish()
                        budget.finish()
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    (DEST / "rag-live-result.json").write_text(json.dumps({"cases": rows, "passed": len(rows) == len(cases) and all(r["passed"] for r in rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
                    if row.get("error_type"):
                        break
    finally:
        Path(path).unlink(missing_ok=True)
        for logger in loggers:
            logger.removeHandler(handler)
        handler.close()
    return len(rows) == 7 and all(r["passed"] for r in rows)


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
