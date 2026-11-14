"""A 阶段开发自检；固定数据与真实模型/商城验收分别报告。"""
import asyncio
import json
import os
import tempfile
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent.budget import BudgetStop, TurnBudget, budget_scope
from agent.nodes.recommend import recommend_node
from agent.nodes.total import total_node
from agent.router import IntentRouter
from agent.tool import mall, product
from agent.tool.schemas import CategoryNav, CategoryRequest, FilterParams, MallError, ProductListRequest, DiscoveryRequest
from tests.mall_fixtures import detail, selection
from tests.check_runtime_boundaries import app, collect


def nav(id=1, name="数码", children=True):
    return CategoryNav(id=id, name=name, level=1, hasChildren=children, sortOrder=0)


class CategoryTransportTests(unittest.TestCase):
    def test_root_and_child_queries_preserve_envelope_and_no_credentials(self):
        with patch.dict(os.environ, {"MALL_BASE_URL": "http://mall.test/api"}), patch.object(mall, "urlopen") as transport:
            response = transport.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = json.dumps({"code": 200, "message": "ok", "data": [nav().model_dump()]}).encode()
            for parent in (None, 1):
                rows = mall.list_categories(CategoryRequest(parentId=parent))
                self.assertEqual(rows, [nav()])
                request = transport.call_args.args[0]
                self.assertEqual(urlsplit(request.full_url).path, "/api/categories")
                self.assertEqual(parse_qs(urlsplit(request.full_url).query), {} if parent is None else {"parentId": ["1"]})
                self.assertEqual(request.header_items(), [])
                self.assertIsNone(request.data)

    def test_invalid_category_response_is_failure_not_empty(self):
        with patch.dict(os.environ, {"MALL_BASE_URL": "http://mall.test"}), patch.object(mall, "urlopen") as transport:
            response = transport.return_value.__enter__.return_value
            response.status = 200
            for data in (None, [{"id": 1}], [{**nav().model_dump(), "hasChildren": 1}], [{**nav().model_dump(), "id": "1"}]):
                with self.subTest(data=data):
                    response.read.return_value = json.dumps({"code": 200, "message": "ok", "data": data}).encode()
                    with self.assertRaises(MallError) as error:
                        mall.list_categories(CategoryRequest())
                    self.assertEqual(error.exception.kind, "response")


class StageNodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.extract = self.enterContext(patch("agent.nodes.recommend.invoke_model", new_callable=AsyncMock, return_value=FilterParams(category="数码")))
        self.enterContext(patch("agent.nodes.recommend.get_extract_client", return_value=MagicMock()))
        self.categories = self.enterContext(patch.object(mall, "list_categories", return_value=[nav(), nav(2, "家电")]))
        self.products = self.enterContext(patch("agent.nodes.recommend.find_products", new_callable=AsyncMock, return_value=selection()))
        self.state = {"messages": [HumanMessage(content="推荐数码商品")], "intent": "商品推荐"}

    async def choose(self, first, query="第二个"):
        state = {**self.state, **first, "messages": [HumanMessage(content=query)]}
        route = IntentRouter(client=AsyncMock())
        state.update(await route.node(state))
        return await total_node(state)

    async def browse(self):
        return await self.choose({}, "浏览分类")

    async def test_terminal_page_survives_repeated_next_requests(self):
        first = await self.browse()
        state = await self.choose(first, "第一个")
        self.extract.return_value = FilterParams(next_page=True)
        self.products.reset_mock()
        for _ in range(2):
            old_page = state["product_page"]
            state = await recommend_node({**state, "messages": [HumanMessage(content="下一批")]})
            self.assertEqual(state["product_page"], old_page)
            self.assertIn("没有下一页", state["final_reply"])
        self.products.assert_not_awaited()

    async def test_failed_next_keeps_list_and_search_cursor_for_explicit_retry(self):
        for mode, cursor, field in (("list", 55, "afterId"), ("search", "opaque-next", "cursor")):
            with self.subTest(mode=mode):
                self.products.side_effect = None
                self.extract.return_value = FilterParams(category="数码")
                if mode == "list":
                    state = await self.choose(await self.browse(), "第一个")
                else:
                    state = await recommend_node(self.state)
                state["product_page"].update(hasMore=True, nextCursor=cursor)
                self.extract.return_value = FilterParams(
                    category=state["task_context"]["filters"]["category"], next_page=True,
                )
                self.products.reset_mock()
                self.products.side_effect = MallError("network")
                failed = await recommend_node({**state, "messages": [HumanMessage(content="下一批")]})
                self.assertEqual(self.products.await_count, 1)
                self.assertEqual(getattr(self.products.await_args.args[0], field), cursor)
                self.assertEqual(failed["product_page"], state["product_page"])
                self.assertEqual(failed["candidate_products"], [])
                self.products.side_effect = None
                await recommend_node({**failed, "messages": [HumanMessage(content="再查下一批")]})
                self.assertEqual(getattr(self.products.await_args.args[0], field), cursor)
                self.assertEqual(self.products.await_count, 2)

    async def test_long_selection_and_browse_ordinals_are_out_of_range(self):
        first = await self.browse()
        for text in ("第" + "9" * 5000 + "个", "查看第" + "9" * 5000 + "个子分类"):
            with self.subTest(command=text[:3]):
                self.assertLess(len(text), 6000)
                result = await self.choose(first, text)
                self.assertIn("不在", result["final_reply"])
                self.assertEqual(result["category_context"], first["category_context"])
        self.products.assert_not_awaited()
        self.categories.assert_called_once()

    async def test_explicit_root_browse_skips_extraction_and_product_search(self):
        result = await self.browse()
        self.extract.assert_not_awaited()
        self.products.assert_not_awaited()
        self.categories.assert_called_once_with(CategoryRequest())
        self.assertEqual([c["id"] for c in result["category_context"]["candidates"]], [1, 2])
        self.assertEqual(result["category_context"]["candidates"][1]["path"], [{"id": 2, "name": "家电"}])
        self.assertIsNone(result["category_context"]["selected"])

    async def test_ordinal_beats_stale_product_and_parent_is_selectable(self):
        first = await self.browse()
        first["candidate_products"] = [{"product_id": 999, "name": "旧商品"}]
        result = await self.choose(first, "第一个")
        request = self.products.await_args.args[0]
        self.assertIsInstance(request, ProductListRequest)
        self.assertEqual(request.categoryId, 1)
        self.assertEqual(result["category_context"]["selected"]["source"], "user")
        self.assertEqual(result["product_page"]["mode"], "list")
        self.categories.assert_called_once()
        self.extract.assert_not_awaited()

    async def test_browse_one_branch_keeps_verified_path_and_does_not_claim_unique(self):
        first = await self.browse()
        self.categories.return_value = [nav(3, "耳机", False)]
        result = await self.choose(first, "查看第二个子分类")
        self.categories.assert_called_with(CategoryRequest(parentId=2))
        candidate = result["category_context"]["candidates"][0]
        self.assertEqual(candidate["path"], [{"id": 2, "name": "家电"}, {"id": 3, "name": "耳机"}])
        self.assertEqual(candidate["parent_id"], 2)
        self.assertTrue(result["category_context"]["pending"])
        self.products.assert_not_awaited()
        self.assertEqual(self.categories.call_count, 2)

    async def test_empty_child_keeps_parent_choice(self):
        first = await self.browse()
        self.categories.return_value = []
        result = await self.choose(first, "查看第一个子分类")
        self.assertEqual(result["category_context"], first["category_context"])
        self.products.assert_not_awaited()
        self.assertIn("子分类为空", result["final_reply"])

    async def test_out_of_range_ordinal_does_not_query(self):
        first = await self.browse()
        result = await self.choose(first, "第99个")
        self.assertIn("不在", result["final_reply"])
        self.products.assert_not_awaited()
        self.categories.assert_called_once()

    async def test_keyword_fallback_is_explicit_and_no_category_is_claimed(self):
        browsing = await self.choose(
            {"task_context": {"filters": FilterParams(category="数码").model_dump()}},
            "浏览分类",
        )
        result = await self.choose(browsing, "按关键词搜索")
        self.assertIsInstance(self.products.await_args.args[0], DiscoveryRequest)
        self.assertIsNone(self.products.await_args.args[0].categoryId)
        self.assertEqual(self.products.await_args.args[0].q, "数码")
        self.assertIn("未确认精确分类", result["final_reply"])
        self.categories.assert_called_once()

    async def test_category_404_reports_category_failure_without_product_lookup(self):
        self.categories.side_effect = MallError("http", http_status=404, code=404)
        result = await self.browse()
        self.assertIn("分类查询未完成", result["final_reply"])
        self.assertNotIn("商品已无法获取", result["final_reply"])
        self.assertNotIn("重新选择商品", result["final_reply"])
        self.products.assert_not_awaited()

    async def test_keyword_search_does_not_query_categories(self):
        result = await recommend_node(self.state)
        self.categories.assert_not_called()
        self.products.assert_awaited_once()
        self.assertEqual(self.products.await_args.args[0], DiscoveryRequest(q="数码", size=3))
        self.assertIn("按关键词搜索", result["final_reply"])

    async def test_empty_root_can_search_but_failure_cannot(self):
        self.categories.return_value = []
        result = await self.browse()
        self.products.assert_awaited_once()
        self.assertIn("按关键词搜索", result["final_reply"])
        self.products.reset_mock()
        self.categories.side_effect = MallError("business", code=404)
        failed = await self.browse()
        self.products.assert_not_awaited()
        self.assertIn("分类查询未完成", failed["final_reply"])
        self.assertEqual(failed["product_page"], {})

    async def test_budget_counts_category_and_prevents_next_query(self):
        self.categories.return_value = []
        self.products.side_effect = product.find_products
        with patch.object(mall, "search_products") as search:
            budget = TurnBudget("a-category-budget")
            budget.MAX_QUERIES = 1
            with budget_scope(budget), self.assertRaisesRegex(BudgetStop, "query_calls"):
                await self.browse()
            self.assertEqual(budget.queries, 1)
            search.assert_not_called()

    async def test_cancel_category_does_not_schedule_products(self):
        async def cancelled(*args):
            raise asyncio.CancelledError()
        with patch("agent.nodes.recommend.invoke_query", side_effect=cancelled):
            with self.assertRaises(asyncio.CancelledError):
                await self.browse()
        self.products.assert_not_awaited()

    async def test_topic_change_exits_pending_selection(self):
        first = await self.browse()
        router = IntentRouter(client=AsyncMock())
        with patch.object(router, "decision", return_value=("闲聊", [])):
            update = await router.node({**first, "messages": [HumanMessage(content="你好，聊聊别的")]})
        self.assertEqual(update["category_context"], {})
        self.assertEqual(update["task_context"], {})
        self.assertEqual(update["category_action"], {})

    async def test_budget_and_color_changes_keep_category_reset_cursor(self):
        first = await self.browse()
        selected = await self.choose(first, "第一个")
        selected["product_page"].update(hasMore=True, nextCursor=55)
        for params in (FilterParams(max_price="300", next_page=True),
                       FilterParams(color="黑色", next_page=True),
                       FilterParams(next_page=True)):
            with self.subTest(params=params):
                old = {**selected, "product_page": {**selected["product_page"], "filters": {**selected["product_page"]["filters"], "color": "旧颜色"}}}
                self.extract.return_value = params
                result = await recommend_node({**old, "messages": [HumanMessage(content="修改条件")]})
                self.assertEqual(result["category_context"]["selected"]["id"], 1)
                self.assertIsNone(self.products.await_args.args[0].afterId)
        self.categories.assert_called_once()

    async def test_same_conditions_page_cursor_and_mode_are_required(self):
        first = await self.browse()
        selected = await self.choose(first, "第一个")
        selected["product_page"].update(hasMore=True, nextCursor=55)
        self.extract.return_value = FilterParams(next_page=True)
        await recommend_node({**selected, "messages": [HumanMessage(content="下一批")]})
        self.assertEqual(self.products.await_args.args[0].afterId, 55)
        selected["product_page"]["mode"] = "search"
        await recommend_node({**selected, "messages": [HumanMessage(content="下一批")]})
        self.assertIsNone(self.products.await_args.args[0].afterId)

    async def test_new_category_clears_selection_and_page(self):
        first = await self.browse()
        selected = await self.choose(first, "第一个")
        selected["product_page"].update(hasMore=True, nextCursor=55)
        self.extract.return_value = FilterParams(category="服装", next_page=True)
        self.products.reset_mock()
        result = await recommend_node({**selected, "messages": [HumanMessage(content="换成服装")]})
        self.assertIsNone(result["category_context"]["selected"])
        self.assertEqual(result["product_page"]["mode"], "search")
        self.assertIsNone(result["product_page"]["category_id"])
        self.categories.assert_called_once()
        self.products.assert_awaited_once()
        self.assertEqual(self.products.await_args.args[0], DiscoveryRequest(q="服装", size=3))

    async def test_unsupported_hard_specs_stop_before_queries(self):
        self.extract.return_value = FilterParams(category="数码", unsupported_specs=["256GB", "专业版"])
        result = await recommend_node(self.state)
        self.categories.assert_not_called()
        self.products.assert_not_awaited()
        self.assertIn("尚不支持", result["final_reply"])
        self.assertEqual(result["task_context"]["filters"]["unsupported_specs"], ["256GB", "专业版"])

    async def test_actual_colors_need_confirmation_then_requery_and_clear_page(self):
        self.categories.return_value = []
        self.extract.return_value = FilterParams(category="数码", color="黑色")
        self.products.return_value = selection().model_copy(update={"alternative_colors": ["曜石黑"]})
        first = await recommend_node(self.state)
        self.assertEqual(first["task_context"]["color_choices"], ["曜石黑"])
        self.assertIn("不自动等同", first["final_reply"])
        self.extract.return_value = FilterParams(category="数码", color="曜石黑", next_page=True)
        self.products.reset_mock()
        blocked = await recommend_node({**first, "messages": [HumanMessage(content="下一批")]})
        self.products.assert_not_awaited()
        self.assertIn("请明确回复", blocked["final_reply"])
        self.products.return_value = selection()
        result = await recommend_node({**first, "messages": [HumanMessage(content="就要曜石黑")]})
        self.assertEqual(result["task_context"]["color_confirmation"], {"requested": "黑色", "value": "曜石黑"})
        self.assertIsNone(self.products.await_args.args[0].cursor)
        self.extract.return_value = FilterParams(category="数码", color=None)
        cancelled = await recommend_node({**result, "messages": [HumanMessage(content="颜色不限")]})
        self.assertNotIn("color_confirmation", cancelled["task_context"])
        self.assertIsNone(cancelled["task_context"]["filters"]["color"])

    async def test_large_level_is_not_silently_truncated(self):
        self.categories.return_value = [nav(i, "长分类名称" * 100) for i in range(1, 30)]
        result = await self.browse()
        self.assertIn("超出", result["final_reply"])
        self.assertEqual(result["category_context"], {})
        self.products.assert_not_awaited()


class ColorEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_alternatives_require_same_sku_price_availability_and_unambiguous_color(self):
        item = detail()
        item.options[0].values[0].value = "曜石黑"
        page = selection(item).page
        with patch.object(mall, "search_products", return_value=page), patch.object(mall, "get_product", return_value=item):
            result = await product.find_products(DiscoveryRequest(), FilterParams(color="黑色", max_price="300"))
            self.assertEqual(result.matches, [])
            self.assertEqual(result.alternative_colors, ["曜石黑"])
            item.skus[0].available = False
            result = await product.find_products(DiscoveryRequest(), FilterParams(color="黑色"))
            self.assertEqual(result.alternative_colors, [])
            item.skus[0].available = True
            result = await product.find_products(DiscoveryRequest(), FilterParams(color="黑色", max_price="100"))
            self.assertEqual(result.alternative_colors, [])
            item.skus[0].optionValueIds = []
            result = await product.find_products(DiscoveryRequest(), FilterParams(color="黑色"))
            self.assertEqual(result.alternative_colors, [])
            self.assertEqual(result.unverified_product_ids, [item.id])


class CategoryPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_restores_category_order_and_other_session_cannot_select_it(self):
        with tempfile.TemporaryDirectory() as directory:
            db = directory + "/stage-a.db"
            model = AsyncMock()
            model.ainvoke.return_value = AIMessage(content=json.dumps({"intent": "商品推荐", "subjects": [], "needs_clarification": False}))
            with app(db) as (main, _, products), patch.object(main.router, "client", model), patch.object(mall, "list_categories", return_value=[nav(), nav(2, "家电")]):
                await collect(main, "浏览分类", "a")
                products.find_products.assert_not_awaited()
            with app(db) as (main, _, products), patch.object(main.router, "client", model):
                await collect(main, "第二个", "a")
                self.assertEqual(products.find_products.await_args.args[0].categoryId, 2)
            async with AsyncSqliteSaver.from_conn_string(db) as saver:
                graph = main.builder.compile(checkpointer=saver)
                restored = await graph.aget_state({"configurable": {"thread_id": "a"}})
                other = await graph.aget_state({"configurable": {"thread_id": "b"}})
                self.assertEqual(restored.values["category_context"]["selected"]["id"], 2)
                self.assertNotIn("category_context", other.values)
