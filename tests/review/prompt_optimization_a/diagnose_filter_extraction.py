"""Observe raw tool arguments for parse failures; no response repair or retries."""
import sys
sys.path.insert(0,'D:/desktop/PythonProject')
import asyncio,json
from pathlib import Path
from langchain_core.messages import HumanMessage
from agent.deps import get_extract_client,INTERNAL_TAG
from agent.memory import memory_manager
from agent.prompts import TOTAL_EXTRACT_SYSTEM
from agent.tool.schemas import FilterParams

async def run():
 rows=[]
 model=get_extract_client().with_structured_output(FilterParams,method='function_calling',include_raw=True).with_config(tags=[INTERNAL_TAG])
 data=json.loads((Path(__file__).parent/'data/extract_probe.json').read_text(encoding='utf8'))
 for i in range(data['attempts']):
  state={'intent':'商品推荐','messages':[HumanMessage(content=data['query'])]}
  result=await model.ainvoke(memory_manager.build_messages(state,TOTAL_EXTRACT_SYSTEM,purpose='extract'))
  raw=result['raw'];parsed=result['parsed'];error=result['parsing_error']
  rows.append({'attempt':i+1,'raw_tool_calls':raw.tool_calls,'parsed':parsed.model_dump(mode='json') if parsed else None,'error':str(error) if error else None,'usage':raw.usage_metadata})
  (OUT/'extract-probe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
  print(json.dumps(rows[-1],ensure_ascii=False),flush=True)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--output-dir',type=Path,required=True)
 args=parser.parse_args()
 OUT=args.output_dir.resolve()
 allowed=Path('D:/desktop/PythonProject/docs/09-验收与交付/证据/提示词优化A/原始证据/审查测试').resolve()
 if OUT.parent != allowed:parser.error('output-dir must be a new direct child of the review evidence directory')
 OUT.mkdir(exist_ok=False)
 asyncio.run(run())
