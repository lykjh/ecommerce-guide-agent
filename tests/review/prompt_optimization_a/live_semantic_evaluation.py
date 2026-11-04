"""Frozen 30 dialogue cases, six groups, three runs; real models, fixed mall data."""
import sys
sys.path.insert(0,'D:/desktop/PythonProject')
import asyncio,json,hashlib,tempfile,time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent import main_agent as main
from agent.tool import mall
from agent.tool.schemas import CategoryNav,ProductDetail,ProductPage,DiscoveryPage,MallError

OUT=None
PREFIX='a-semantic'
# Each turn carries expected state/behavior; all turns must pass for a case to pass.
CASES=[]
FIXTURES=json.loads((Path(__file__).parent/'data/mall_fixtures.json').read_text(encoding='utf8'))


def product(pid=101,injected=False):
 return ProductDetail.model_validate(FIXTURES['injected_product'] if injected else FIXTURES['products'][str(pid)])

async def run():
 report={'scope':'real models + fixed HTTP-boundary fixtures; full dialogue sequences; not production prevalence estimate','plan_sha256':hashlib.sha256(json.dumps(CASES,ensure_ascii=False).encode()).hexdigest(),'cases':CASES,'rounds':[]}
 (OUT/(PREFIX+'-plan.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 with tempfile.TemporaryDirectory(prefix='a-semantic-') as directory,patch.object(main,'DB_PATH',str(Path(directory)/'isolated.db')):
  for round_no in range(1,4):
   results=[];report['rounds'].append({'round':round_no,'results':results})
   for group,name,queries,golds in CASES:
    row={'group':group,'id':name,'turns':[],'passed':True};results.append(row)
    calls=[]
    def categories(req):
     calls.append({'kind':'category','args':req.model_dump(mode='json')})
     if name=='category_404':raise MallError('http',http_status=404,code=404)
     rows=FIXTURES['roots'] if req.parentId is None else [(FIXTURES['child_ids'][str(req.parentId)],'耳机',False)]
     return [CategoryNav(id=i,name=n,level=1 if req.parentId is None else 2,hasChildren=h,sortOrder=k) for k,(i,n,h) in enumerate(rows)]
    def listing(req):
     calls.append({'kind':'list','args':req.model_dump(mode='json')})
     if name=='product_503':raise MallError('http',http_status=503,code=503)
     if name=='response_invalid':raise MallError('response')
     items=[] if name=='empty' else [FIXTURES['listing_item']]
     return ProductPage(items=items,size=3,hasMore=False,nextCursor=None)
    def search(req):
     page=listing(req);calls[-1]['kind']='search'
     return DiscoveryPage(items=page.items,hasMore=False,nextCursor=None)
    def detail(pid):
     calls.append({'kind':'detail','id':pid})
     if pid not in (101,102):raise MallError('http',http_status=404,code=404)
     return product(pid,name=='description')
    with ExitStack() as stack:
     for key,fn in [('list_categories',categories),('list_products',listing),('search_products',search),('get_product',detail)]:stack.enter_context(patch.object(mall,key,fn))
     for turn,(query,gold) in enumerate(zip(queries,golds)):
      calls.clear();start=time.monotonic(); item={'query':query,'expected':gold};row['turns'].append(item)
      try:
       reply=''.join([x async for x in main.get_ai_stream(query,f'{round_no}-{group}-{name}')])
       async with AsyncSqliteSaver.from_conn_string(main.DB_PATH) as saver:
        state=(await main.builder.compile(checkpointer=saver).aget_state({'configurable':{'thread_id':f'{round_no}-{group}-{name}'}})).values
       category=state.get('category_context',{}); filters=state.get('task_context',{}).get('filters',{})
       checks={}
       for k,v in gold.items():
        if k in ('max_price','color'):checks[k]=filters.get(k)==v
        elif k=='pending':checks[k]=category.get('pending')==v
        elif k=='not_pending':checks[k]=not category.get('pending')
        elif k=='selected':checks[k]=(category.get('selected') or {}).get('id')==v
        elif k=='path':checks[k]=all(len(c['path'])==v for c in category.get('candidates',[])) and bool(category.get('candidates'))
        elif k=='matches':checks[k]='SKU ' in reply
        elif k=='no_matches':checks[k]='SKU ' not in reply
        elif k=='no_query':checks[k]=not calls
        elif k=='search':checks[k]=any(c['kind']=='search' for c in calls)
        elif k=='sku':checks[k]=f'SKU {v}' in reply
        elif k=='price':checks[k]=v+' 元' in reply
        elif k=='unsupported':checks[k]=bool(filters.get('unsupported_specs'))==v and (not v or '尚不支持' in reply)
        elif k=='intent':checks[k]=state.get('intent')==v
        elif k=='detail':checks[k]=any(c.get('id')==v for c in calls)
        elif k=='details':checks[k]=all(any(c.get('id')==p for c in calls) for p in v)
        elif k=='contains':checks[k]=v in reply
        elif k=='not_contains':checks[k]=v not in reply
        elif k=='failure':checks[k]=any(x in reply for x in ['未完成','失败','异常','不可用']) and '没有找到' not in reply
        elif k=='unavailable':checks[k]=any(x in reply for x in ['无货','缺货','不可售','售罄','没有库存','暂无库存','暂时无货','不可购买','无库存'])
        elif k=='unknown':checks[k]=any(x in reply for x in ['未提供','未标注','没有提供','没有标注','无法确认','未说明','没有说明','未明确','未注明','不确定','未提及','没有相关','未包含'])
       if name=='category_404':checks['category_called']=any(c['kind']=='category' for c in calls)
       item.update(reply=reply,calls=list(calls),checks=checks,intent=state.get('intent'),filters=filters,passed=all(checks.values()))
      except Exception as e:item.update(error_type=type(e).__name__,passed=False,calls=list(calls))
      item['seconds']=round(time.monotonic()-start,3);row['passed'] &= item['passed']
      (OUT/(PREFIX+'-three.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'round':round_no,'group':group,'id':name,'passed':row['passed']},ensure_ascii=False),flush=True)
   report['rounds'][-1]['passed']=sum(r['passed'] for r in results)
   report['rounds'][-1]['by_group']={g:{'passed':sum(r['passed'] for r in results if r['group']==g),'total':sum(c[0]==g for c in CASES)} for g in sorted({c[0] for c in CASES})}
   (OUT/(PREFIX+'-three.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 print('THREE_ROUNDS_COMPLETE',flush=True)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--cases',choices=['semantic_cases','pagination_holdout'],default='semantic_cases')
 args=parser.parse_args()
 allowed=Path('D:/desktop/PythonProject/docs/09-验收与交付/证据/提示词优化A/原始证据/审查测试').resolve()
 OUT=args.output_dir.resolve()
 if OUT.parent != allowed:parser.error('output-dir must be a new direct child of the review evidence directory')
 OUT.mkdir(exist_ok=False)
 CASES=json.loads((Path(__file__).parent/'data'/f'{args.cases}.json').read_text(encoding='utf8'))
 PREFIX=args.cases
 asyncio.run(run())
