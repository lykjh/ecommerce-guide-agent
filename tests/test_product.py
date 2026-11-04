"""商品业务离线测试：HTTP 为替身，执行真实查询预算入口。"""
import unittest
from decimal import Decimal
from unittest.mock import patch, call

from agent.budget import BudgetStop, TurnBudget, budget_scope
from agent.tool import product
from agent.tool.schemas import (CandidateRef, DiscoveryPage, DiscoveryRequest, FilterParams,
    MallError, ProductCard, ProductDetail, ProductListRequest, ProductPage)


class ProductTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.detail = ProductDetail.model_validate({
            "id": 101, "name": "耳机甲", "description": None,
            "options": [{"id": 1, "name": "颜色", "values": [
                {"id": 11, "value": "黑色"}, {"id": 12, "value": "白色"}]}],
            "skus": [
                {"id": 1001, "optionValueIds": [11], "price": "299.99", "available": True},
                {"id": 1002, "optionValueIds": [12], "price": "199", "available": True},
                {"id": 1003, "optionValueIds": [11], "price": "599", "available": True}]})
        self.card = ProductCard(id=101, name="耳机甲", startingPrice="199", soldOut=False)
        self.page = DiscoveryPage(items=[self.card], hasMore=False, nextCursor=None)
        self.request = DiscoveryRequest(q="耳机", maxPrice="500")
        self.filters = FilterParams(category="耳机", max_price="500", color="黑色")

    async def test_same_sku_match_and_evidence_excludes_other_skus(self):
        with patch.object(product.mall, "search_products", return_value=self.page) as search, \
             patch.object(product.mall, "get_product", return_value=self.detail) as detail:
            result = await product.find_products(self.request, self.filters)
        search.assert_called_once_with(self.request)
        detail.assert_called_once_with(101)
        self.assertEqual([s.id for s in result.matches[0].skus], [1001])
        self.assertEqual([s.id for s in result.matches[0].product.skus], [1001])
        self.assertEqual(result.checked_product_ids, [101])
        self.assertEqual(result.unverified_product_ids, [])
        self.assertEqual(len(self.detail.skus), 3)

    async def test_starting_price_cannot_substitute_for_color_price(self):
        detail = self.detail.model_copy(update={"skus": self.detail.skus[1:]})
        with patch.object(product.mall, "search_products", return_value=self.page), \
             patch.object(product.mall, "get_product", return_value=detail):
            result = await product.find_products(self.request, self.filters)
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unverified_product_ids, [])

    async def test_missing_color_or_unknown_option_semantics_are_unverified(self):
        for options in ([], [self.detail.options[0].model_copy(update={"name": "款式"})],
                        [self.detail.options[0], self.detail.options[0]]):
            with self.subTest(options=options), \
                 patch.object(product.mall, "search_products", return_value=self.page), \
                 patch.object(product.mall, "get_product", return_value=self.detail.model_copy(update={"options": options})):
                result = await product.find_products(self.request, self.filters)
            self.assertEqual(result.matches, [])
            self.assertEqual(result.unverified_product_ids, [101])

    async def test_missing_sku_color_relation_is_unverified(self):
        sku = self.detail.skus[0].model_copy(update={"optionValueIds": []})
        with patch.object(product.mall, "search_products", return_value=self.page), \
             patch.object(product.mall, "get_product", return_value=self.detail.model_copy(update={"skus": [sku]})):
            result = await product.find_products(self.request, self.filters)
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unverified_product_ids, [101])

    async def test_cancelled_color_does_not_require_color_evidence(self):
        filters = FilterParams(max_price="500", color=None)
        with patch.object(product.mall, "search_products", return_value=self.page), \
             patch.object(product.mall, "get_product", return_value=self.detail.model_copy(update={"options": []})):
            result = await product.find_products(self.request, filters)
        self.assertEqual([s.id for s in result.matches[0].skus], [1001, 1002])
        self.assertEqual(result.unverified_product_ids, [])

    async def test_unavailable_sku_is_not_recommended(self):
        sku = self.detail.skus[0].model_copy(update={"available": False})
        with patch.object(product.mall, "search_products", return_value=self.page), \
             patch.object(product.mall, "get_product", return_value=self.detail.model_copy(update={"skus": [sku]})):
            result = await product.find_products(self.request, self.filters)
        self.assertEqual(result.matches, [])

    async def test_decimal_bounds_are_inclusive_and_exact(self):
        filters = FilterParams(min_price="299.99", max_price="300.00")
        skus = [self.detail.skus[0].model_copy(update={"id": i+1, "price": Decimal(price)})
                for i, price in enumerate(["299.98", "299.99", "300.00", "300.01"])]
        with patch.object(product.mall, "search_products", return_value=self.page), \
             patch.object(product.mall, "get_product", return_value=self.detail.model_copy(update={"skus": skus})):
            result = await product.find_products(self.request, filters)
        self.assertEqual([s.price for s in result.matches[0].skus], [Decimal("299.99"), Decimal("300.00")])

    async def test_one_page_three_details_and_remaining_unverified(self):
        page = DiscoveryPage(items=[self.card.model_copy(update={"id": n}) for n in range(101, 106)],
                             hasMore=True, nextCursor="opaque-A")
        with patch.object(product.mall, "search_products", return_value=page) as search, \
             patch.object(product.mall, "get_product", side_effect=lambda n: self.detail.model_copy(update={"id": n})) as details:
            result = await product.find_products(self.request, self.filters)
        search.assert_called_once()
        self.assertEqual(details.call_args_list, [call(101), call(102), call(103)])
        self.assertEqual(result.checked_product_ids, [101, 102, 103])
        self.assertEqual(result.unverified_product_ids, [104, 105])
        self.assertTrue(result.page.hasMore)
        self.assertEqual(result.page.nextCursor, "opaque-A")

    async def test_list_uses_numeric_cursor_and_correct_endpoint(self):
        page = ProductPage(items=[], size=3, hasMore=True, nextCursor=101)
        request = ProductListRequest(categoryId=663, afterId=200)
        with patch.object(product.mall, "list_products", return_value=page) as listing, \
             patch.object(product.mall, "search_products") as search, \
             patch.object(product.mall, "get_product") as detail:
            result = await product.find_products(request, self.filters)
        listing.assert_called_once_with(request)
        search.assert_not_called()
        detail.assert_not_called()
        self.assertIsInstance(result.page.nextCursor, int)
        self.assertEqual(result.matches, [])

    async def test_each_http_call_counts_and_budget_stop_propagates(self):
        page = self.page.model_copy(update={"items": [self.card, self.card.model_copy(update={"id": 102})]})
        budget = TurnBudget("product-budget")
        budget.MAX_QUERIES = 2
        with patch.object(product.mall, "search_products", return_value=page), \
             patch.object(product.mall, "get_product", return_value=self.detail) as detail, budget_scope(budget):
            with self.assertRaisesRegex(BudgetStop, "query_calls"):
                await product.find_products(self.request, self.filters)
        self.assertEqual(budget.queries, 2)
        detail.assert_called_once_with(101)

    async def test_upstream_failure_is_not_empty_result(self):
        error = MallError("http", http_status=503, code=503)
        with patch.object(product.mall, "search_products", side_effect=error) as search:
            with self.assertRaises(MallError) as caught:
                await product.find_products(self.request, self.filters)
        self.assertIs(caught.exception, error)
        search.assert_called_once()

    async def test_known_candidate_name_and_id_resolve_without_search(self):
        candidate = CandidateRef(product_id=101, name="耳机甲")
        for subject in ("耳机甲", "101", "商品编号:101", "商品编号：101"):
            with self.subTest(subject=subject), \
                 patch.object(product.mall, "search_products") as search, \
                 patch.object(product.mall, "get_product", return_value=self.detail) as detail:
                result = await product.resolve_product(subject, [candidate])
            search.assert_not_called()
            detail.assert_called_once_with(101)
            self.assertEqual(result.status, "resolved")
            self.assertEqual(result.product.id, 101)

    async def test_direct_explicit_id_does_not_need_candidates(self):
        with patch.object(product.mall, "search_products") as search, \
             patch.object(product.mall, "get_product", return_value=self.detail) as detail:
            result = await product.resolve_product("101", [])
        search.assert_not_called()
        detail.assert_called_once_with(101)
        self.assertEqual(result.status, "resolved")

    async def test_known_duplicate_names_require_selection(self):
        candidates = [CandidateRef(product_id=n, name="耳机甲") for n in (101, 102)]
        with patch.object(product.mall, "search_products") as search, patch.object(product.mall, "get_product") as detail:
            result = await product.resolve_product("耳机甲", candidates)
        search.assert_not_called()
        detail.assert_not_called()
        self.assertEqual(result.status, "choose")
        self.assertEqual(result.candidates, candidates)

    async def test_unique_exact_name_on_complete_page_is_confirmed(self):
        with patch.object(product.mall, "search_products", return_value=self.page) as search, \
             patch.object(product.mall, "get_product", return_value=self.detail) as detail:
            result = await product.resolve_product("耳机甲", [])
        self.assertEqual(search.call_args.args[0].q, "耳机甲")
        detail.assert_called_once_with(101)
        self.assertEqual(result.status, "resolved")

    async def test_similar_names_and_incomplete_page_require_selection(self):
        pages = [self.page.model_copy(update={"hasMore": True, "nextCursor": "next"}),
                 self.page.model_copy(update={"items": [self.card.model_copy(update={"name": "耳机甲升级版"})]}),
                 self.page.model_copy(update={"items": [self.card, self.card.model_copy(update={"id": 102})]})]
        for page in pages:
            with self.subTest(page=page), patch.object(product.mall, "search_products", return_value=page), \
                 patch.object(product.mall, "get_product") as detail:
                result = await product.resolve_product("耳机甲", [])
            detail.assert_not_called()
            self.assertEqual(result.status, "choose")

    async def test_empty_search_is_not_found(self):
        with patch.object(product.mall, "search_products", return_value=self.page.model_copy(update={"items": []})), \
             patch.object(product.mall, "get_product") as detail:
            result = await product.resolve_product("耳机甲", [])
        self.assertEqual(result.status, "not_found")
        detail.assert_not_called()

    async def test_detail_404_only_is_translated_to_not_found(self):
        with patch.object(product.mall, "get_product", side_effect=MallError("http", http_status=404, code=404)):
            result = await product.resolve_product("101", [])
        self.assertEqual(result.status, "not_found")
        error = MallError("http", http_status=503, code=503)
        with patch.object(product.mall, "get_product", side_effect=error):
            with self.assertRaises(MallError) as caught:
                await product.resolve_product("101", [])
        self.assertIs(caught.exception, error)

    async def test_search_error_is_not_translated_to_not_found(self):
        with patch.object(product.mall, "search_products", side_effect=MallError("http", http_status=404, code=404)):
            with self.assertRaises(MallError):
                await product.resolve_product("耳机甲", [])

    async def test_query_decimal_is_not_converted_to_float(self):
        request = DiscoveryRequest(q="耳机", maxPrice="0.10")
        with patch.object(product.mall,"search_products",return_value=DiscoveryPage(items=[],hasMore=False,nextCursor=None)) as search:
            await product.find_products(request, FilterParams(max_price="0.10"))
        self.assertEqual(search.call_args.args[0].maxPrice, Decimal("0.10"))
        self.assertIsInstance(search.call_args.args[0].maxPrice, Decimal)
