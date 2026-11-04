"""冻结期望值后测试真实模型，不修改生产代码或按结果改写期望。"""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import collect_runs
from agent.router import APP_CFG, IntentRouter


async def main():
    root = Path(__file__).resolve().parents[1]
    holdout = "--holdout" in sys.argv
    source = Path(__file__).with_name("router_holdout_cases.json" if holdout else "router_boundary_cases.json")
    cases = json.loads(source.read_text(encoding="utf-8"))
    recheck = "--repeat-failures" in sys.argv
    if recheck:
        previous = json.loads((root / "docs" / "09-验收与交付" / "证据" / "专项测试" / "router-boundary-live-result-before-fix.json").read_text(encoding="utf-8"))
        failures = {row["id"] for row in previous["results"] if not row["passed"]}
        cases = [case for case in cases if case["id"] in failures] * 3
    router = IntentRouter()
    rows = []
    report = {"run_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": "constructed, not real user traffic", "model": APP_CFG.get("llm.router_model"),
              "cases_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256((root / "agent" / "prompts.py").read_bytes()).hexdigest(),
              "planned": len(cases), "results": rows}
    destination = root / "docs" / "09-验收与交付" / "证据" / "专项测试" / ("router-holdout-result.json" if holdout else "router-boundary-recheck.json" if recheck else "router-boundary-live-result.json")
    for case in cases:
        messages = [HumanMessage(content=text) if index % 2 == 0 else AIMessage(content=text)
                    for index, text in enumerate(case["history"])] + [HumanMessage(content=case["query"])]
        started = time.monotonic()
        with collect_runs() as captured:
            actual = await router.router({"messages": messages, "summary": case["summary"]})
        row = {"id": case["id"], "group": case["group"], "expected": case["expected"],
               "actual": actual, "passed": actual == case["expected"],
               "query_chars": len(case["query"]), "seconds": round(time.monotonic() - started, 3)}
        rows.append(row)
        if not row["passed"]:
            row["model_outputs"] = [generation[0]["text"] for run in captured.traced_runs
                                    if run.run_type == "llm" and run.outputs
                                    for generation in [run.outputs.get("generations", [[]])[0]] if generation]
        if not row["passed"] or len(rows) % 20 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if len(rows) >= 3 and all(r["actual"] == "识别失败" for r in rows[-3:]):
            report["stopped"] = "three consecutive model failures"
            break
    report["by_group"] = {group: {"total": count, "passed": sum(r["passed"] for r in rows if r["group"] == group)}
                          for group, count in Counter(r["group"] for r in rows).items()}
    report["passed_count"] = sum(row["passed"] for row in rows)
    report["complete"] = len(rows) == len(cases)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "passed": report["passed_count"], "total": len(rows), "groups": report["by_group"]}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
