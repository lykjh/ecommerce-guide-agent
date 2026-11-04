"""冻结多轮边界样本的结构及实体来源复测，显式执行才调用模型。"""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.router import IntentRouter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import collect_runs


async def main():
    case = next(c for c in json.loads(Path(__file__).with_name("router_boundary_cases.json").read_text(encoding="utf-8"))
                if c["id"] == "multi_turn-08")
    state = {"messages": [HumanMessage(content=t) if i % 2 == 0 else AIMessage(content=t) for i, t in enumerate(case["history"])]
                         + [HumanMessage(content=case["query"])], "summary": case["summary"]}
    rows = []
    router = IntentRouter()
    for repeat in range(3):
        with collect_runs() as captured:
            result = await router.node(state)
        row = {"repeat": repeat, "intent": result["intent"], "subjects": result["route_subjects"],
               "passed": result["intent"] == case["expected"] and result["route_subjects"] == ["保温杯甲", "保温杯乙"]}
        row["raw"] = [generation[0]["text"] for run in captured.traced_runs if run.run_type == "llm" and run.outputs
                      for generation in [run.outputs.get("generations", [[]])[0]] if generation]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    destination = Path(__file__).resolve().parents[1] / "docs/06-问题发现与解决/2026-11-01/证据/rag-schema-final-recheck.json"
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return all(row["passed"] for row in rows)


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
