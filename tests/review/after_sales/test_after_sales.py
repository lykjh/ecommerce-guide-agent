import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from agent.identity import Identity
from agent.nodes.orders import OrderTurns, OrderSelection
from agent.nodes.aftersales import guidance, STEPS
from agent.tool.orders import OrderAccess, OrderAuthExpired, OrderError, OrderDetail, AfterSalePage, order_after_sales


def detail(status="PAID", id=101):
    return OrderDetail(id=id, shopName="测试店", status=status, totalAmount="20.00", pointUnits=0, items=[])


def case(status="REQUESTED", id=501, orderId=101, refund=None):
    return dict(id=id, orderId=orderId, type="RETURN_REFUND", status=status, refund=refund,
                reason="PRIVATE_REASON", returnDestination={"address":"PRIVATE_ADDRESS"})


def page(items=(), cursor=None):
    return AfterSalePage(items=list(items), nextBeforeId=cursor)


class GuidanceTests(unittest.TestCase):
    def test_states_have_correct_next_step_and_no_private_fields(self):
        for status, step in STEPS.items():
            with self.subTest(status=status):
                result=guidance(detail(), page([case(status)]), "售后进度")
                self.assertIn(step, result)
                self.assertNotIn("PRIVATE_", result)
                self.assertIn("不退真实资金", result)
    def test_empty_is_not_submitted(self):
        result=guidance(detail(), page(), "退货")
        for word in ("退货退款", "未查到", "7天", "法定例外", "没有提交"):
            self.assertIn(word,result)
    def test_exchange_has_no_fake_entry(self):
        result=guidance(detail(),page(),"换货")
        self.assertIn("没有在线换货申请入口",result)
        self.assertNotIn("选择换货",result)
    def test_unpaid_cancelled_not_eligible(self):
        for status in ("UNPAID","CANCELLED"):
            result=guidance(detail(status),page(),"售后")
            self.assertNotIn("选择仅退款",result)
    def test_refund_unknown_and_failed_not_success(self):
        for status, word in (("UNKNOWN","结果未确定"),("FAILED","失败")):
            result=guidance(detail(),page([case("REFUNDING",refund={"status":status})]),"进度")
            self.assertIn(word,result)
            self.assertNotIn("已到账",result)
    def test_pagination_does_not_claim_exhaustive(self):
        result=guidance(detail(),page([case(id=n) for n in range(505,500,-1)],501),"售后进度")
        self.assertIn("最近5条",result)
    def test_policy_failure_not_fabricated(self):
        with patch("agent.nodes.aftersales.load_policy",side_effect=ValueError):
            self.assertIn("政策资料暂不可用",guidance(detail(),page(),"退货"))


class TurnTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.turns=OrderTurns()
        self.owner=Identity("user","A","login-A")
        self.access=OrderAccess(self.owner,"http://mall.invalid","private-cookie","secret",MagicMock())
        self.access.identity.resolve=AsyncMock(return_value=self.owner)
        self.d=patch("agent.nodes.orders.order_detail",new_callable=AsyncMock,return_value=detail()).start()
        self.p=patch("agent.nodes.orders.order_after_sales",new_callable=AsyncMock,return_value=page()).start()
        self.addCleanup(patch.stopall)
    async def test_order_then_followup_refreshes(self):
        self.assertIn("没有提交",await self.turns.handle("订单101退货怎么申请", "s",self.access))
        self.p.return_value=page([case("WAITING_RETURN")])
        self.assertIn("等待寄回",await self.turns.handle("进度呢", "s",self.access))
        self.assertEqual(self.p.await_count,2)
        self.assertEqual(self.access.identity.resolve.await_count,2)
    async def test_list_ordinal_and_detail_then_refund(self):
        from time import monotonic
        self.turns.sessions["s"]=OrderSelection(self.owner,monotonic()+60,(101,102))
        self.assertIn("订单 101",await self.turns.handle("第一笔退货", "s",self.access))
        self.assertIn("订单 101",await self.turns.handle("退款条件", "s",self.access))
    async def test_missing_order_and_guest_no_reads(self):
        for query in ("我要退货","售后进度","售后"):
            self.assertIn("请先输入",await self.turns.handle(query,"s",self.access))
        guest=OrderAccess(Identity("guest"),self.access.base,None,"secret",self.access.identity)
        self.assertIn("请先登录",await self.turns.handle("订单101退款", "g",guest))
        self.d.assert_not_awaited()
        self.p.assert_not_awaited()
    async def test_other_identity_cannot_reuse_selection(self):
        await self.turns.handle("订单101售后","s",self.access)
        other=OrderAccess(Identity("user","B","login-B"),self.access.base,"B-cookie","secret",self.access.identity)
        self.d.reset_mock()
        self.assertIn("请先输入",await self.turns.handle("售后进度","s",other))
        self.d.assert_not_awaited()
    async def test_identity_changes_after_read_discard_reply(self):
        self.access.identity.resolve.return_value=Identity("user","B","login-B")
        with self.assertRaises(OrderAuthExpired):
            await self.turns.handle("订单101售后","s",self.access)
        self.assertNotIn("s",self.turns.sessions)
    async def test_invalid_and_ambiguous_do_not_reuse_previous(self):
        await self.turns.handle("订单101售后","s",self.access)
        self.d.reset_mock()
        for query in ("订单abc售后", "订单101和102售后", "订单0售后", "订单999999999999999999999999售后", "朋友的订单101售后"):
            await self.turns.handle(query,"s",self.access)
        self.d.assert_not_awaited()
    async def test_service_failure_not_empty(self):
        self.p.side_effect=OrderError("unavailable")
        result=await self.turns.handle("订单101售后","s",self.access)
        self.assertIn("暂不可查询",result)
        self.assertNotIn("未查到",result)
    async def test_public_policy_and_navigation_unchanged(self):
        self.assertIsNone(await self.turns.handle("七日无理由政策", "s",self.access))
        await self.turns.handle("订单101售后","s",self.access)
        self.assertEqual(await self.turns.handle("返回导购","s",self.access),"已返回商品导购。")
        self.assertNotIn("s",self.turns.sessions)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_get_parsing_cookie_and_ownership(self):
        access=OrderAccess(Identity("user","A","fp"),"http://mall.invalid","cookie-A","secret",MagicMock())
        for payload, valid in (({"items":[case()],"nextBeforeId":None},True),
                               ({"items":[case(orderId=102)],"nextBeforeId":None},False),
                               ({"items":[case(status="UNRECOGNIZED")],"nextBeforeId":None},False),
                               ({"items":[case(),case()],"nextBeforeId":None},False),
                               ({"items":[],"nextBeforeId":500},False)):
            response=MagicMock(status=200)
            response.__enter__.return_value=response
            response.read.return_value=json.dumps({"code":200,"data":payload}).encode()
            opener=MagicMock()
            opener.open.return_value=response
            with patch("agent.tool.orders.build_opener",return_value=opener):
                if valid:
                    result=await order_after_sales(access,101)
                    self.assertEqual(result.items[0].orderId,101)
                    self.assertNotIn("PRIVATE",result.model_dump_json())
                else:
                    with self.assertRaises(OrderError): await order_after_sales(access,101)
                request=opener.open.call_args.args[0]
                self.assertEqual(request.get_method(),"GET")
                self.assertEqual(request.full_url,"http://mall.invalid/orders/101/after-sales?size=5")
                self.assertEqual(request.get_header("Cookie"),"MALLAUTH=cookie-A")

if __name__=="__main__": unittest.main(verbosity=2)
