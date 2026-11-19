"""Independent ASGI/SSE dialogue checks; local mall fixture; optional real configured model.
Run: python -B regression.py offline|live OUTPUT_DIRECTORY
Only synthetic identity/order data; no production writes or browser acceptance.
"""
import asyncio
from contextlib import ExitStack
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from threading import Thread
from unittest.mock import AsyncMock, patch
import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
with patch('agent.utils.log_tool.get_loger', side_effect=logging.getLogger):
    from agent import http_api, main_agent
from agent.identity import Identity
from agent.nodes.orders import OrderTurns
from agent.deps import APP_CFG

MODE, OUT = sys.argv[1], Path(sys.argv[2])
assert MODE in ('offline', 'live') and OUT.is_dir()
logging.basicConfig(filename=OUT / (MODE + '.log'), level=logging.INFO, encoding='utf-8')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Each tuple fixes the question and required answer before execution.
OFFLINE = {
 'canonical': [('我的订单','101'),('第二笔退货怎么申请','订单 102'),('办理条件','7天'),('办理入口','售后服务'),('进度呢','等待寄回'),('换货呢','没有在线换货申请入口')],
 'natural_followups': [('订单101退货怎么申请','订单 101'),('在哪里申请？','售后服务'),('需要满足哪些条件？','7天'),('现在办到哪了？','待审核'),('能帮我直接办理吗？','不代为提交')],
 'order_switch': [('我的订单','102'),('第一笔退货','订单 101'),('第二笔退款','订单 102'),('进度呢','订单 102'),('订单101售后','订单 101'),('进度呢','订单 101')],
 'failed_switch': [('订单101售后','订单 101'),('订单999售后','未查到可访问'),('进度呢','请.*(?:选择|订单编号|我的订单)')],
 'ambiguous_switch': [('订单101售后','订单 101'),('订单101和102售后','订单编号'),('进度呢','请.*(?:选择|订单编号|我的订单)')],
 'explicit_public_policy': [('订单101售后','订单 101'),('不问这笔了，商城一般的退货运费政策是什么？','PUBLIC_GRAPH')],
 'unpaid': [('订单103退货','无需申请退款'),('换货呢','没有在线换货申请入口')],
 'cancelled': [('订单104退款','订单已取消'),('进度呢','不能据此认定已退款到账')],
 'failure_recovery': [('订单101售后','订单 101'),('进度呢','暂不可查询'),('进度呢','等待寄回')],
 'ownership': [('订单101售后','订单 101'),('朋友的订单102售后','只能查看'),('进度呢','IDENTITY_REJECTED')],
 'fresh_progress': [('订单101售后','待审核'),('进度呢','等待寄回'),('进度呢','退款处理中'),('进度呢','不能证明真实资金已到账')],
}
LIVE = {
 'invoice': [('可以自动开电子发票吗？','第12条'),('那需要多久？','具备开票条件且所需资料完整后.*7个自然日'),('那帮我直接开好吧','不能|不代|无法|不支持')],
 'freight': [('退货运费谁承担？','第9条'),('如果是质量问题呢？','经营者承担.*必要运输费用'),('那不喜欢了呢？','消费者|买家')],
 'mixed': [('七天无理由退货有什么条件？','第6.1条'),('我的订单','101'),('第二笔退货怎么申请','订单 102'),('在哪里申请？','售后服务'),('现在办到哪了？','等待寄回'),('能换货吗？','没有在线换货申请入口')],
 'policy_after_order': [('订单101售后','订单 101'),('不问这笔了，商城一般的退货运费政策是什么？','第9条')],
}

requests = []
fixture = {'status':'REQUESTED','fail':False}
class Mall(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        requests.append({'method':'GET','path':self.path,'cookie_ok':self.headers.get('Cookie')=='MALLAUTH=controlled-A'})
        code, data = 200, None
        def order(n):
            return dict(id=n,shopName='受控店铺',status={103:'UNPAID',104:'CANCELLED'}.get(n,'PAID'), totalAmount='20.00',pointUnits=0)
        if self.path == '/orders?size=5':
            data=dict(items=[dict(**order(n),itemCount=0) for n in (101,102)],hasMore=False,nextCursor=None)
        elif match := re.fullmatch(r'/orders/(10[1-4])',self.path):
            data=dict(**order(int(match[1])),items=[])
        elif match := re.fullmatch(r'/orders/(10[1-4])/after-sales\?size=5',self.path):
            n=int(match[1]); status='WAITING_RETURN' if n==102 else fixture['status']
            data=dict(items=[] if n>=103 else [dict(id=n+400,orderId=n,type='RETURN_REFUND',status=status,refund={'status':'SUCCESS'} if status=='REFUNDED' else None)],nextBeforeId=None)
            if fixture['fail']: code=503
        else: code=404
        body=json.dumps({'code':code,'data':data}).encode()
        self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)

async def sentinel(*args,**kwargs):
    yield 'PUBLIC_GRAPH'

usages=[]
class Usage(main_agent.TurnUsage):
    def finish(self):
        usages.append(dict(calls=self.calls,input_tokens=self.input_tokens,output_tokens=self.output_tokens,unknown=self.unknown_usage,failures=self.failures))
        super().finish()

def hashes():
    paths=list((ROOT/'agent').rglob('*.py'))+list((ROOT/'agent/rag/policies').glob('*'))+[ROOT/'config/app.yml']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file()}

async def run():
    before=hashes(); rows=[]
    (OUT/(MODE+'-source-before.json')).write_text(json.dumps(before,indent=2),encoding='utf-8')
    server=ThreadingHTTPServer(('127.0.0.1',0),Mall); worker=Thread(target=server.serve_forever,daemon=True); worker.start()
    try:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp, ExitStack() as stack:
            database=str(Path(temp)/'isolated.db')
            stack.enter_context(patch.dict(os.environ,{'AGENT_SESSION_SECRET':'review-multiturn-secret-20261116-long','AGENT_PUBLIC_ORIGIN':'http://review','MALL_BASE_URL':f'http://127.0.0.1:{server.server_port}'}))
            stack.enter_context(patch.object(http_api,'DB_PATH',database)); stack.enter_context(patch.object(main_agent,'DB_PATH',database))
            stack.enter_context(patch.object(http_api,'order_turns',OrderTurns()))
            stack.enter_context(patch.object(main_agent,'TurnUsage',Usage))
            if MODE=='offline': stack.enter_context(patch.object(main_agent,'_get_ai_stream',sentinel))
            async with http_api.app.router.lifespan_context(http_api.app):
                identity=stack.enter_context(patch.object(http_api.app.state.identity,'resolve',new_callable=AsyncMock,return_value=Identity('user','701','a'*64)))
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http_api.app),base_url='http://review',headers={'Origin':'http://review'},cookies={'MALLAUTH':'controlled-A'}) as client:
                    for round_no in range(1,4 if MODE=='live' else 2):
                        for name, sequence in (LIVE if MODE=='live' else OFFLINE).items():
                            identity.return_value=Identity('user','701','a'*64)
                            fixture.update(status='REQUESTED',fail=False)
                            session=await client.post('/api/sessions'); assert session.status_code==201
                            headers={'Authorization':'Bearer '+session.json()['session_token']}
                            for index,(question,expected) in enumerate(sequence):
                                if name=='failure_recovery': fixture.update(fail=index==1,status='WAITING_RETURN')
                                if name=='fresh_progress': fixture['status']=['REQUESTED','WAITING_RETURN','REFUNDING','REFUNDED'][index]
                                if name=='ownership' and index==2: identity.return_value=Identity('user','702','b'*64)
                                offset=len(requests)
                                response=await client.post('/api/chat',headers=headers,json={'message':question})
                                frames=[]
                                if response.status_code==200:
                                    for block in response.text.strip().split('\n\n'):
                                        fields=dict(line.split(':',1) for line in block.splitlines())
                                        frames.append((fields['event'].strip(),json.loads(fields['data'])))
                                reply=''.join(d['text'] for e,d in frames if e=='delta')
                                checks={'expected':bool(re.search(expected,reply)), 'sse':bool(frames) and frames[-1][0]=='done' and all(e!='error' for e,d in frames),'only_read':all(r['method']=='GET' and r['cookie_ok'] for r in requests[offset:])}
                                if expected=='IDENTITY_REJECTED': checks.update(expected=response.status_code==409,sse=not frames)
                                if name in ('failed_switch','ambiguous_switch') and index==2: checks['no_old_order']='订单 101' not in reply and not requests[offset:]
                                if name=='fresh_progress': checks['refetched']=len(requests[offset:])==2
                                row=dict(round=round_no,scenario=name,turn=index+1,question=question,expected=expected,status=response.status_code,reply=reply,events=[e for e,d in frames],requests=requests[offset:],checks=checks,passed=all(checks.values()))
                                rows.append(row)
                                (OUT/(MODE+'-results.json')).write_text(json.dumps(dict(model=APP_CFG.get('llm.router_model') if MODE=='live' else 'not called',cases=rows,usage=usages),ensure_ascii=False,indent=2),encoding='utf-8')
                                print(f'{MODE} round={round_no} {name}/{index+1} pass={row["passed"]}',flush=True)
    finally:
        server.shutdown(); server.server_close(); worker.join()
        after=hashes(); (OUT/(MODE+'-source-after.json')).write_text(json.dumps(after,indent=2),encoding='utf-8')
    result=dict(mode=MODE,total=len(rows),passed=sum(r['passed'] for r in rows),source_stable=before==after,model_calls=sum(u['calls'] for u in usages),tokens=sum(u['input_tokens']+u['output_tokens'] for u in usages),unknown_usage=sum(u['unknown'] for u in usages),browser='NOT_EXECUTED')
    (OUT/(MODE+'-summary.json')).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
    return all(r['passed'] for r in rows) and before==after

if __name__=='__main__': raise SystemExit(0 if asyncio.run(run()) else 1)
