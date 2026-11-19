"""Strict, read-only requests to the configured mall on behalf of one login."""

from dataclasses import dataclass, field
from decimal import Decimal
from http.client import HTTPException
import json
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from agent.budget import invoke_query
from agent.identity import Identity, IdentityClient, MALL_AUTH_COOKIE, NoRedirect


OrderId = Annotated[int, Field(strict=True, gt=0, le=2**63 - 1)]
Count = Annotated[int, Field(strict=True, ge=0)]
Money = Annotated[Decimal, Field(ge=0, max_digits=18, decimal_places=2, allow_inf_nan=False)]
PAGE_SIZE = 5


class OrderError(Exception):
    pass


class OrderAuthExpired(OrderError):
    pass


@dataclass(frozen=True)
class OrderAccess:
    owner: Identity
    base: str
    cookie: str | None = field(repr=False)
    secret: str = field(repr=False)
    identity: IdentityClient = field(repr=False)

    async def verify(self):
        if await self.identity.resolve(self.cookie, self.base, self.secret) != self.owner:
            raise OrderAuthExpired


class OrderSummary(BaseModel):
    id: OrderId
    shopName: str | None
    status: Literal["UNPAID", "PAID", "CANCELLED"]
    totalAmount: Money
    pointUnits: Count


class OrderRow(OrderSummary):
    itemCount: Count


class OrderPage(BaseModel):
    items: list[OrderRow]
    hasMore: Annotated[bool, Field(strict=True)]
    nextCursor: OrderId | None


class OrderItem(BaseModel):
    productName: str
    spec: str
    quantity: Annotated[int, Field(strict=True, gt=0)]


class OrderDetail(OrderSummary):
    items: list[OrderItem]


class RefundProgress(BaseModel):
    status: Literal["PENDING", "UNKNOWN", "SUCCESS", "FAILED"]


class AfterSale(BaseModel):
    id: OrderId
    orderId: OrderId
    type: Literal["REFUND_ONLY", "RETURN_REFUND"]
    status: Literal["REQUESTED", "REJECTED", "WAITING_RETURN", "WAITING_RECEIPT",
                    "REFUNDING", "REFUNDED", "CLOSED"]
    refund: RefundProgress | None


class AfterSalePage(BaseModel):
    items: list[AfterSale]
    nextBeforeId: OrderId | None


class Parcel(BaseModel):
    shipmentId: OrderId
    orderId: OrderId
    status: Literal["SHIPPED", "CONFIRMED"]
    carrierName: str | None
    trackingNo: str | None = Field(exclude=True, repr=False)
    shippedAt: str | None
    confirmedAt: str | None


def _get(access, path, result_type):
    if access.owner.kind != "user" or not access.cookie:
        raise OrderAuthExpired
    try:
        request = Request(access.base.rstrip("/") + path, headers={
            "Cookie": f"{MALL_AUTH_COOKIE}={access.cookie}", "Cache-Control": "no-cache",
        })
        try:
            response = build_opener(NoRedirect()).open(request, timeout=4)
        except HTTPError as error:
            response = error
        with response:
            status = response.status
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise OrderError("unavailable")
        body = json.loads(raw, parse_float=Decimal)
        if not isinstance(body, dict) or type(body.get("code")) is not int or body["code"] != status:
            raise OrderError("unavailable")
        if status == 401:
            raise OrderAuthExpired
        if status != 200:
            raise OrderError({400: "invalid", 403: "forbidden", 404: "missing"}.get(status, "unavailable"))
        return result_type.validate_python(body.get("data"))
    except (URLError, OSError, HTTPException, ValidationError, ValueError):
        raise OrderError("unavailable") from None


async def list_orders(access: OrderAccess, after_id: int | None):
    params: dict[str, int] = {"size": PAGE_SIZE}
    if after_id is not None:
        params["afterId"] = TypeAdapter(OrderId).validate_python(after_id)
    page = await invoke_query(_get, access, "/orders?" + urlencode(params), TypeAdapter(OrderPage))
    ids = [item.id for item in page.items]
    if len(ids) != len(set(ids)) or len(ids) > PAGE_SIZE or (
        page.hasMore and (not ids or page.nextCursor != ids[-1])
    ) or (not page.hasMore and page.nextCursor is not None) or after_id in ids:
        raise OrderError("unavailable")
    return page


async def order_detail(access: OrderAccess, order_id: int):
    order_id = TypeAdapter(OrderId).validate_python(order_id)
    detail = await invoke_query(_get, access, f"/orders/{order_id}", TypeAdapter(OrderDetail))
    if detail.id != order_id:
        raise OrderError("unavailable")
    return detail


async def order_parcels(access: OrderAccess, order_id: int):
    order_id = TypeAdapter(OrderId).validate_python(order_id)
    parcels = await invoke_query(_get, access, f"/orders/{order_id}/shipments", TypeAdapter(list[Parcel]))
    if any(parcel.orderId != order_id for parcel in parcels) or len({p.shipmentId for p in parcels}) != len(parcels):
        raise OrderError("unavailable")
    return parcels


async def order_after_sales(access: OrderAccess, order_id: int):
    order_id = TypeAdapter(OrderId).validate_python(order_id)
    page = await invoke_query(_get, access, f"/orders/{order_id}/after-sales?size={PAGE_SIZE}",
                              TypeAdapter(AfterSalePage))
    ids = [item.id for item in page.items]
    if (len(ids) > PAGE_SIZE or len(ids) != len(set(ids))
            or any(item.orderId != order_id for item in page.items)
            or (page.nextBeforeId is not None and (len(ids) != PAGE_SIZE or page.nextBeforeId != ids[-1]))):
        raise OrderError("unavailable")
    return page
