"""Explicit live regression: project model, one attempt per case, fail fast."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from agent import router
from agent.deps import APP_CFG
from agent.budget import TurnBudget, budget_scope

import argparse
import re
from datetime import datetime
DATA = Path(__file__).resolve().parent / "data"
_parser = argparse.ArgumentParser(description="Explicit live evaluation using project model configuration; consumes quota.")
_parser.add_argument("--output-dir", type=Path, required=True, help="New YYYYMMDD-HHMMSS-id directory directly under the developer evidence directory")
_args = _parser.parse_args()
OUT = _args.output_dir.resolve()
_evidence = Path(__file__).resolve().parents[3] / "docs/09-验收与交付/证据/开发/提示词优化A/原始证据"
if OUT.parent != _evidence.resolve() or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-.+", OUT.name):
 _parser.error("Output must be a new timestamped run directory under " + str(_evidence))
OUT.mkdir(exist_ok=False)
with (_evidence.parent / "README.md").open("a", encoding="utf8") as _handoff:
 _handoff.write("\n运行登记：" + datetime.now().isoformat(timespec="seconds") + " · " + str(Path(__file__).resolve()) + " · [" + str(OUT) + "](" + OUT.as_posix() + ")（登记不代表通过）\n")

def state(query, **kwargs):
    history = kwargs.pop('history', [])
    return dict(messages=history + [HumanMessage(content=query)], **kwargs)

CASES = [(name, {**state, "messages": [(HumanMessage if m["role"]=="human" else AIMessage)(content=m["content"]) for m in state["messages"]]}, intent, set(subjects) if subjects is not None else None) for name,state,intent,subjects in json.loads((DATA/"route_boundaries.json").read_text(encoding="utf8"))]

class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.events = []
    def on_llm_end(self, response, **kwargs):
        message = response.generations[0][0].message
        self.events.append({'raw':message.content,'usage':message.usage_metadata})
    def on_llm_error(self, error, **kwargs):
        self.events.append({'error_type':type(error).__name__})

async def main():
    original = router.invoke_model
    rows = []
    for name, incoming, intent, subjects in CASES:
        recorder = Recorder()
        async def measured(*args, **kwargs):
            kwargs['config'] = {**kwargs.get('config', {}), 'callbacks':[recorder]}
            return await original(*args, **kwargs)
        with patch.object(router, 'invoke_model', side_effect=measured), budget_scope(TurnBudget(name)):
            actual, found = await router.IntentRouter().decision(incoming)
        passed = actual == intent and not any('error_type' in e for e in recorder.events)
        if subjects is not None:
            passed = passed and (set(found) == subjects if intent=='商品对比' else len(found)==1 and found[0] in subjects)
        rows.append({'case':name,'query':incoming['messages'][-1].content,'expected_intent':intent,'intent':actual,'subjects':found,'passed':passed,'events':recorder.events})
        (OUT/'route-boundaries.json').write_text(json.dumps({'model':APP_CFG.get('llm.router_model'),'cases':rows},ensure_ascii=False,indent=2),encoding='utf8')
        print(name, passed, flush=True)
        if not passed:
            raise SystemExit('STOP: first failed route boundary')

if __name__=='__main__':
    assert len(CASES) == 12
    asyncio.run(main())
