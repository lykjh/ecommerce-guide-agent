import asyncio
from decimal import Decimal
from io import BytesIO
import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from agent.budget import BudgetStop, TurnBudget, budget_scope, invoke_query
from agent.tool.mall import get_product, list_products, search_products
from agent.tool.schemas import DiscoveryRequest, MallError, ProductListRequest


class MallTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MALL_BASE_URL": "http://mall.test/api/"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.transport = patch("agent.tool.mall.urlopen")
        self.open = self.transport.start()
        self.addCleanup(self.transport.stop)
        self.response = self.open.return_value.__enter__.return_value
        self.response.status = 200

    def test_search_serializes_decimal_and_opaque_cursor(self):
        self.response.read.return_value = b'{"code":200,"message":"success","data":{"items":[],"hasMore":false,"nextCursor":null}}'
        result = search_products(DiscoveryRequest(q="黑色耳机", maxPrice=Decimal("0.10"), size=20, cursor="a+/=&"))
        request = self.open.call_args.args[0]
        self.assertEqual(urlsplit(request.full_url).path, "/api/discovery/products")
        self.assertEqual(parse_qs(urlsplit(request.full_url).query),
                         {"q": ["黑色耳机"], "maxPrice": ["0.10"], "size": ["20"], "cursor": ["a+/=&"]})
        self.assertEqual(request.method, "GET")
        self.assertIsNone(request.data)
        self.assertEqual(request.header_items(), [])
        self.assertEqual(result.items, [])
        self.assertIsNone(result.nextCursor)
        self.assertNotIn("total", result.model_dump())
        self.open.assert_called_once()

    def test_list_uses_integer_cursor(self):
        self.response.read.return_value = b'{"code":200,"message":"success","data":{"items":[],"size":3,"hasMore":true,"nextCursor":88}}'
        result = list_products(ProductListRequest(categoryId=7, afterId=90))
        request = self.open.call_args.args[0]
        self.assertEqual(urlsplit(request.full_url).path, "/api/products")
        self.assertEqual(parse_qs(urlsplit(request.full_url).query),
                         {"categoryId": ["7"], "afterId": ["90"], "size": ["3"]})
        self.assertEqual(result.nextCursor, 88)
        self.assertIs(type(result.nextCursor), int)

    def test_detail_decimal_and_nullable_description(self):
        self.response.read.return_value = b'{"code":200,"message":"success","data":{"id":101,"name":"A","description":null,"options":[],"skus":[{"id":1,"optionValueIds":[],"price":0.10,"available":true}]}}'
        result = get_product(101)
        self.assertEqual(self.open.call_args.args[0].full_url, "http://mall.test/api/products/101")
        self.assertEqual(result.skus[0].price, Decimal("0.10"))
        self.assertIsNone(result.description)

    def test_card_without_sku_can_have_null_starting_price(self):
        self.response.read.return_value = b'{"code":200,"message":"success","data":{"items":[{"id":1,"name":"A","startingPrice":null,"soldOut":true}],"hasMore":false,"nextCursor":null}}'
        result = search_products(DiscoveryRequest())
        self.assertIsNone(result.items[0].startingPrice)
        self.assertTrue(result.items[0].soldOut)

    def test_http_errors_preserve_status_and_never_retry(self):
        for status in (400, 404, 429, 503):
            with self.subTest(status=status):
                self.open.reset_mock()
                body = json.dumps({"code": status, "message": "secret-response", "data": None}).encode()
                self.open.side_effect = HTTPError("http://mall.test/private", status, "secret", {}, BytesIO(body))
                with self.assertRaises(MallError) as caught:
                    get_product(101)
                self.assertEqual((caught.exception.kind, caught.exception.http_status, caught.exception.code),
                                 ("http", status, status))
                self.assertEqual(str(caught.exception), "http")
                self.open.assert_called_once()

    def test_http_error_non_json_keeps_http_status(self):
        self.open.side_effect = HTTPError("http://mall.test", 503, "secret", {}, BytesIO(b'<html>secret</html>'))
        with self.assertRaises(MallError) as caught:
            get_product(101)
        self.assertEqual(caught.exception.http_status, 503)
        self.assertIsNone(caught.exception.code)
        self.assertEqual(caught.exception.kind, "http")

    def test_success_http_with_business_failure(self):
        self.response.read.return_value = b'{"code":503,"message":"secret","data":null}'
        with self.assertRaises(MallError) as caught:
            search_products(DiscoveryRequest())
        self.assertEqual((caught.exception.kind, caught.exception.http_status, caught.exception.code), ("business", 200, 503))

    def test_invalid_response_does_not_become_empty_result(self):
        for body in (b'not json', b'{"code":true,"message":"ok","data":{}}',
                     b'{"code":200,"message":"ok","data":{"items":"bad","hasMore":false,"nextCursor":null}}',
                     b'{"code":200,"message":"ok","data":{"items":[],"hasMore":false,"nextCursor":7}}'):
            with self.subTest(body=body):
                self.response.read.return_value = body
                with self.assertRaises(MallError) as caught:
                    search_products(DiscoveryRequest())
                self.assertEqual(caught.exception.kind, "response")

    def test_network_errors_are_classified_without_body(self):
        for error in (URLError("secret network"), ConnectionResetError("secret reset")):
            with self.subTest(error=type(error).__name__):
                self.open.side_effect = error
                with self.assertRaises(MallError) as caught:
                    get_product(101)
                self.assertEqual(str(caught.exception), "network")
                self.assertIsNone(caught.exception.code)

    def test_timeout_propagates_and_does_not_retry(self):
        for error in (TimeoutError("secret"), URLError(TimeoutError("secret"))):
            with self.subTest(error=type(error).__name__):
                self.open.reset_mock()
                self.open.side_effect = error
                with self.assertRaises(TimeoutError):
                    get_product(101)
                self.open.assert_called_once()

    def test_transport_timeout_respects_remaining_budget(self):
        self.response.read.return_value = b'{"code":200,"message":"ok","data":{"items":[],"hasMore":false,"nextCursor":null}}'
        budget = TurnBudget("http-budget")
        budget.deadline = time.monotonic() + budget.ANSWER_SECONDS + 0.5
        with budget_scope(budget):
            search_products(DiscoveryRequest())
        self.assertGreater(self.open.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(self.open.call_args.kwargs["timeout"], 0.5)
        self.assertEqual(budget.queries, 0)
        budget.deadline = time.monotonic() - 1
        self.open.reset_mock()
        with budget_scope(budget), self.assertRaises(TimeoutError):
            search_products(DiscoveryRequest())
        self.open.assert_not_called()

    def test_environment_address_changes_without_reimport(self):
        self.response.read.return_value = b'{"code":200,"message":"ok","data":{"items":[],"hasMore":false,"nextCursor":null}}'
        with patch.dict(os.environ, {"MALL_BASE_URL": "https://next.test/v2"}):
            search_products(DiscoveryRequest())
        self.assertEqual(urlsplit(self.open.call_args.args[0].full_url).netloc, "next.test")
        self.assertEqual(urlsplit(self.open.call_args.args[0].full_url).path, "/v2/discovery/products")
        with patch.dict(os.environ, {"MALL_BASE_URL": ""}), self.assertRaises(MallError) as caught:
            search_products(DiscoveryRequest())
        self.assertEqual(caught.exception.kind, "configuration")

    def test_log_does_not_include_query_or_error_body(self):
        body = b'{"code":503,"message":"SECRET_COOKIE SECRET_KEY","data":null}'
        self.open.side_effect = HTTPError("http://mall.test", 503, "secret", {}, BytesIO(body))
        with self.assertLogs("agent.tool.mall", level="INFO") as logs:
            with self.assertRaises(MallError):
                search_products(DiscoveryRequest(q="PRIVATE_QUERY"))
        output = "\n".join(logs.output)
        self.assertIn("endpoint=/discovery/products", output)
        self.assertIn("status=503", output)
        for secret in ("SECRET_COOKIE", "SECRET_KEY", "PRIVATE_QUERY", "http://mall.test"):
            self.assertNotIn(secret, output)


class MallBudgetIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_timeout_becomes_budget_stop(self):
        with patch.dict(os.environ, {"MALL_BASE_URL": "http://mall.test"}), \
             patch("agent.tool.mall.urlopen", side_effect=TimeoutError()), \
             budget_scope(TurnBudget("mall-timeout")):
            with self.assertRaisesRegex(BudgetStop, "query_timeout"):
                await invoke_query(get_product, 101)
