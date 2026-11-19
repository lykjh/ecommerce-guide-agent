"""Local HTTP mall fixture + real Agent HTTP/SSE; no model or business writes."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Thread
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tests import test_http_api as base_tests
events=base_tests.events
from agent import http_api, main_agent
from agent.identity import Identity

class HttpAfterSalesTests(unittest.TestCase):
    setUp=base_tests.HttpApiTests.setUp
    def test_multiturn_real_transport_and_identity(self):
        observed=[]
        fixture={"status":"REQUESTED", "fail":False}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                observed.append((self.command,self.path,self.headers.get("Cookie")))
                code=200
                if self.path=="/orders/101":
                    data=dict(id=101,shopName="受控订单",status="PAID",totalAmount="20.00",pointUnits=0,items=[])
                elif self.path=="/orders/101/after-sales?size=5":
                    data={"items":[dict(id=501,orderId=101,type="RETURN_REFUND",status=fixture["status"],refund=None)],"nextBeforeId":None}
                    if fixture["fail"]: code=503
                else: code,data=404,None
                raw=json.dumps({"code":code,"data":data}).encode()
                self.send_response(code)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with patch.dict("os.environ",{"MALL_BASE_URL":f"http://127.0.0.1:{server.server_port}"}), patch.object(http_api.app.state.identity,"resolve",new_callable=AsyncMock,return_value=Identity("user","701","a"*64)) as identity, patch.object(main_agent,"_get_ai_stream") as graph:
                self.client.cookies.set("MALLAUTH","controlled-A")
                token=self.client.post("/api/sessions").json()["session_token"]
                headers={"Authorization":"Bearer "+token}
                results=[]
                for question,expected in (("订单101退货怎么申请","待审核"),("进度呢","等待寄回"),("换货呢","没有在线换货申请入口")):
                    if question=="进度呢": fixture["status"]="WAITING_RETURN"
                    response=self.client.post("/api/chat",headers=headers,json={"message":question})
                    frames=events(response.text)
                    self.assertEqual([f[0] for f in frames],["meta","delta","done"])
                    reply=frames[1][1]["text"]
                    self.assertIn(expected,reply)
                    results.append({"question":question,"reply":reply})
                fixture["fail"]=True
                response=self.client.post("/api/chat",headers=headers,json={"message":"进度呢"})
                self.assertIn("暂不可查询",events(response.text)[1][1]["text"])
                response=self.client.post("/api/chat",headers=headers,json={"message":"订单999售后"})
                self.assertIn("未查到可访问",events(response.text)[1][1]["text"])
                identity.return_value=Identity("user","702","b"*64)
                response=self.client.post("/api/chat",headers=headers,json={"message":"进度呢"})
                self.assertNotEqual(response.status_code,200)
                graph.assert_not_called()
                self.assertTrue(observed)
                self.assertTrue(all(method=="GET" and cookie=="MALLAUTH=controlled-A" for method,path,cookie in observed))
                print(json.dumps({"controlled_http_dialogue":results,"requests":len(observed),"switched_identity_status":response.status_code},ensure_ascii=False))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

if __name__=="__main__": unittest.main(verbosity=2)
