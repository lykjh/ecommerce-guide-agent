"""Format verified order reads outside the chat graph and model context."""

from collections import OrderedDict
from dataclasses import dataclass, replace
from decimal import Decimal
import re
from time import monotonic

from agent.identity import Identity, IdentityError
from agent.tool.orders import OrderAccess, OrderAuthExpired, OrderError, list_orders, order_detail, order_parcels
from agent.tool.orders import order_after_sales
from agent.nodes.aftersales import guidance


_DETAIL = re.compile(r"(?:查询|查|查看)?\s*(?:我的)?订单(?:编号|号)?\s*[:：#]?\s*([0-9]{1,19})(?:\s*(?:的)?(物流|快递|包裹|(?:现在)?发货了吗))?")
_ORDINAL = re.compile(r"(?:选(?:择)?)?第([一二三四五1-5])(?:个|条|笔)?(?:的)?(物流|快递|包裹)?")
_NEXT = {"下一页", "下一批", "更多订单"}
_LIST = re.compile(r"(?:帮我)?(?:查询|查看|查)?(?:一下)?(?:我的订单|订单列表)|查看订单|查订单|(?:能)?(?:帮我)?(?:看看|查查|查看)最近买的几单(?:吗)?")
_LOGISTICS = {"物流", "查物流", "我的物流", "这单的物流", "这单物流", "查看物流", "这单的快递", "发货了吗", "到哪儿了"}
_PRIVATE = re.compile(r"订单|下单记录|物流|运单|包裹|快递|最近买的几单|我之前买的东西到哪(?:儿|里)了")
_LEAVE = re.compile(r"^(?:退出订单|返回导购|推荐|帮我挑|找商品|比较|对比|浏览分类)")
_WRITE = re.compile(r"退款|取消订单|确认收货|改地址|修改地址|支付订单|催发货")
_POLICY = re.compile(r"政策|无理由|退货|退款|运费|包邮|保修|质保|发票|开票|价保|赔付|服务时限")
_STATUS = {"PAID": "已支付", "UNPAID": "待支付", "CANCELLED": "已取消",
           "SHIPPED": "已发货", "CONFIRMED": "已确认收货"}
_ERRORS = {"missing": "未查到可访问的订单。", "invalid": "订单查询条件无效，请重新选择。",
           "forbidden": "无法访问该订单。", "unavailable": "商城订单服务暂不可查询，请稍后再试。"}
_HELP = "请回复订单编号、当前页序号，或输入「我的订单」查看列表。请勿发送密码、验证码或收货信息。"
_AFTER = re.compile(r"售后|退款|退货|换货")
_AFTER_FOLLOW = re.compile(r"^(?:那|这单)?(?:现在)?(?:进度|到哪(?:一步|儿|里)?了?|办到哪了|怎么办|怎么操作|办理入口|办理条件|在哪里申请|在哪弄|需要满足哪些条件|能帮我直接办理吗|刷新|状态)(?:呢|啊|啦|吗)?$")


@dataclass(frozen=True)
class OrderSelection:
    owner: Identity
    expires_at: float = 0
    ids: tuple[int, ...] = ()
    next_cursor: int | None = None
    selected: int | None = None
    after_sales: bool = False


def _summary(item):
    amount = f"应付 ¥{item.totalAmount:.2f}"
    if item.pointUnits:
        amount += f" + {Decimal(item.pointUnits) / 100:f} 积分"
    return f"订单 {item.id}｜{item.shopName or '商城店铺'}｜{_STATUS[item.status]}｜{amount}"


class OrderTurns:
    def __init__(self):
        self.sessions: OrderedDict[str, OrderSelection] = OrderedDict()

    def _selection(self, session_id, owner):
        now = monotonic()
        while self.sessions and next(iter(self.sessions.values())).expires_at <= now:
            self.sessions.popitem(last=False)
        saved = self.sessions.get(session_id)
        return saved if saved and saved.owner == owner else None

    async def handle(self, message: str, session_id: str, access: OrderAccess):
        query = message.strip().rstrip("。？！?!")
        previous = self._selection(session_id, access.owner)
        if re.match(r"^(?:不问这笔(?:订单)?了|退出订单|返回导购)", query):
            self.sessions.pop(session_id, None)
            return "已返回商品导购。" if query in {"退出订单", "返回导购"} else None
        after_sales = bool((_AFTER.search(query) and (
            _PRIVATE.search(query) or (previous and (previous.selected or re.search(r"第[一二三四五1-5](?:个|条|笔)", query)))
            or query in {"售后", "退款进度", "退货进度", "换货进度"}
            or re.search(r"我的|我要|我想|售后进度|售后状态|申请售后", query)))
            or (previous and previous.after_sales and _AFTER_FOLLOW.fullmatch(query)))
        # Public rules may leave order selection; explicit order references stay private.
        if not after_sales and _POLICY.search(query) and not _PRIVATE.search(query):
            self.sessions.pop(session_id, None)
            return None
        if _LEAVE.match(query) and not _PRIVATE.search(query.removeprefix("退出订单")):
            self.sessions.pop(session_id, None)
            return "已返回商品导购。" if query in {"退出订单", "返回导购"} else None

        if previous is None and not _PRIVATE.search(query) and not after_sales:
            return None
        if access.owner.kind != "user":
            return "请先登录商城，再查询本人订单或物流。"
        if _WRITE.search(query) and not after_sales:
            return "小暖只提供本人订单和物流查询。请到商城订单页办理此操作。"

        state = previous or OrderSelection(access.owner)
        # A failed or ambiguous switch must not leave the old order selected.
        if (re.search(r"订单(?:编号|号)?\s*[:：#]?\s*[A-Za-z0-9+-]|第[一二三四五1-5](?:个|条|笔)", query)
                or re.fullmatch(r"[0-9]+", query)):
            state = replace(state, selected=None, after_sales=after_sales, expires_at=monotonic() + 900)
            self.sessions[session_id] = state
            self.sessions.move_to_end(session_id)
            if len(self.sessions) > 1024:
                self.sessions.popitem(last=False)
        try:
            if after_sales:
                reply, updated = await self._after(query, state, access)
            else:
                reply, updated = await self._read(query, replace(state, after_sales=False), access)
            if updated is not None:
                await access.verify()
                self.sessions[session_id] = replace(updated, expires_at=monotonic() + 900)
                self.sessions.move_to_end(session_id)
                if len(self.sessions) > 1024:
                    self.sessions.popitem(last=False)
            elif previous is None:
                # Keep ambiguous follow-ups in the private branch so they do not reach the model.
                self.sessions[session_id] = replace(state, expires_at=monotonic() + 900)
                if len(self.sessions) > 1024:
                    self.sessions.popitem(last=False)
            return reply
        except OrderAuthExpired:
            self.sessions.pop(session_id, None)
            raise
        except IdentityError:
            return _ERRORS["unavailable"]
        except OrderError as exc:
            return _ERRORS.get(str(exc), _ERRORS["unavailable"])

    async def _after(self, query, state, access):
        if re.search(r"别人|他人|其他人|朋友的", query):
            return "只能查看当前登录账号本人的订单售后。", None
        matches = re.findall(r"订单(?:编号|号)?\s*[:：#]?\s*([0-9]+)", query)
        ordinal = re.search(r"第([一二三四五1-5])(?:个|条|笔)", query)
        order_id = state.selected
        if matches:
            if len(matches) != 1 or len(matches[0]) > 19 or len(re.findall(r"[0-9]+", query)) > 1:
                return _HELP, None
            order_id = int(matches[0])
        elif ordinal:
            index = "一二三四五".index(ordinal[1]) if not ordinal[1].isascii() else int(ordinal[1]) - 1
            order_id = state.ids[index] if index < len(state.ids) else None
        elif re.search(r"订单(?:编号|号)?\s*[:：#]?\s*[A-Za-z0-9]", query):
            return _HELP, None
        if order_id is None or not 0 < order_id < 2**63:
            return "请先输入「我的订单」并选择一笔，或输入「订单编号 + 售后」。我会核对本人订单的办理入口和进度。", None
        order = await order_detail(access, order_id)
        page = await order_after_sales(access, order_id)
        return guidance(order, page, query), replace(state, selected=order_id, after_sales=True)

    async def _read(self, query, state, access):
        if _LIST.fullmatch(query) or query in _NEXT:
            if query in _NEXT and not state.next_cursor:
                return ("已到本次查询末页。" if state.ids else _HELP), None
            page = await list_orders(access, state.next_cursor if query in _NEXT else None)
            items = [f"{n}. {_summary(item)}｜{item.itemCount} 种商品"
                     for n, item in enumerate(page.items, 1)]
            if not items:
                reply = "暂无订单。"
            else:
                ending = "回复「下一页」继续查看。" if page.hasMore else "已到本次查询末页。"
                reply = "\n".join([*items, ending, "可回复当前页序号查看详情。"])
            return reply, replace(state, ids=tuple(item.id for item in page.items),
                                  next_cursor=page.nextCursor, selected=None)

        order_id, logistics = state.selected, query in _LOGISTICS
        if match := _DETAIL.fullmatch(query):
            order_id = int(match[1])
            logistics = bool(match[2])
        elif match := _ORDINAL.fullmatch(query):
            index = "一二三四五".index(match[1]) if not match[1].isascii() else int(match[1]) - 1
            order_id = state.ids[index] if index < len(state.ids) else None
            logistics = bool(match[2])
        elif re.fullmatch(r"[0-9]{1,19}", query):
            order_id = int(query)
        elif not logistics:
            return _HELP, None
        if order_id is None or not 0 < order_id < 2**63:
            return _HELP, None
        if logistics:
            parcels = await order_parcels(access, order_id)
            lines = [f"订单 {order_id} 的包裹"]
            for n, parcel in enumerate(parcels, 1):
                tracking = ("****" + parcel.trackingNo[-4:]) if parcel.trackingNo and len(parcel.trackingNo) > 4 else "****"
                lines.append(f"{n}. {_STATUS[parcel.status]}｜{parcel.carrierName or '承运商未提供'}｜运单 {tracking}")
                if parcel.shippedAt:
                    lines.append(f"发货时间 {parcel.shippedAt}")
                if parcel.confirmedAt:
                    lines.append(f"确认收货时间 {parcel.confirmedAt}")
            return "\n".join(lines if parcels else ["暂无包裹信息。"]), replace(state, selected=order_id)

        order = await order_detail(access, order_id)
        lines = [_summary(order), *(f"- {item.productName}｜{item.spec} × {item.quantity}" for item in order.items)]
        return "\n".join(lines), replace(state, selected=order_id)
