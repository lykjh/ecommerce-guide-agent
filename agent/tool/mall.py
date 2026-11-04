"""公开商城只读 HTTP 适配；每次调用的预算由商品编排负责。"""
import json
import os
import time
from decimal import Decimal
from http.client import HTTPException
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, RootModel, ValidationError

from agent.budget import CURRENT_BUDGET, TurnBudget
from agent.tool.schemas import (
    DiscoveryPage, DiscoveryRequest, MallError, ProductDetail,
    ProductListRequest, ProductPage, CategoryNav, CategoryRequest,
)
from agent.utils.config_tool import get_app_config
from agent.utils.log_tool import get_loger

logger = get_loger("agent.tool.mall")
Result = TypeVar("Result", bound=BaseModel)


class _Envelope(BaseModel):
    code: int = Field(strict=True)
    message: str
    data: Any


def _get(path: str, result_type: type[Result], *, query=None, endpoint: str) -> Result:
    base = os.getenv(get_app_config().get("mall.base_url_env", "MALL_BASE_URL"), "").strip()
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise MallError("configuration")
    url = base.rstrip("/") + path
    if query is not None:
        url += "?" + urlencode(query.model_dump(mode="json", exclude_none=True))
    budget = CURRENT_BUDGET.get()
    timeout = TurnBudget.QUERY_SECONDS
    if budget is not None:
        timeout = min(timeout, budget.deadline - time.monotonic()
                      - (0 if budget.answer_started else budget.ANSWER_SECONDS))
    if timeout <= 0:
        raise TimeoutError("mall request budget exhausted")

    started = time.monotonic()
    status, code, outcome = None, None, "network"
    try:
        try:
            try:
                response = urlopen(Request(url, method="GET"), timeout=timeout)
            except HTTPError as exc:
                response = exc
            with response as response:
                status = response.status
                body = response.read()
        except TimeoutError:
            outcome = "timeout"
            raise
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                outcome = "timeout"
                raise TimeoutError("mall request timed out") from None
            raise MallError("network") from None
        except (OSError, HTTPException):
            raise MallError("network") from None

        if status >= 400:
            outcome = "http"
            try:
                code = _Envelope.model_validate(json.loads(body, parse_float=Decimal)).code
            except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
                # HTTP 状态本身仍可信；错误正文不符合契约时不伪造业务码。
                code = None
            raise MallError("http", http_status=status, code=code) from None

        outcome = "response"
        try:
            envelope = _Envelope.model_validate(json.loads(body, parse_float=Decimal))
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            raise MallError("response", http_status=status) from None
        code = envelope.code
        if code != 200:
            outcome = "business"
            raise MallError("business", http_status=status, code=code)
        try:
            result = result_type.model_validate(envelope.data)
        except ValidationError:
            raise MallError("response", http_status=status, code=code) from None
        outcome = "completed"
        return result
    finally:
        logger.info("mall_request request_id=%s endpoint=%s status=%s code=%s outcome=%s duration_ms=%d",
                    budget.request_id if budget else "standalone", endpoint, status, code, outcome,
                    (time.monotonic() - started) * 1000)


def search_products(request: DiscoveryRequest) -> DiscoveryPage:
    return _get("/discovery/products", DiscoveryPage, query=request, endpoint="/discovery/products")


def list_products(request: ProductListRequest) -> ProductPage:
    return _get("/products", ProductPage, query=request, endpoint="/products")


def get_product(product_id: int) -> ProductDetail:
    return _get(f"/products/{product_id}", ProductDetail, endpoint="/products/{id}")


def list_categories(request: CategoryRequest) -> list[CategoryNav]:
    return _get("/categories", RootModel[list[CategoryNav]], query=request, endpoint="/categories").root
