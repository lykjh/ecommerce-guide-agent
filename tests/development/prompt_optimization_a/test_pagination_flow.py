"""Full router -> graph -> query pagination regression; models and mall are local fixtures."""
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent.router import IntentRouter
from agent.tool import mall
from agent.tool.schemas import FilterParams, CategoryNav, ProductPage, MallError
from tests.check_runtime_boundaries import app, collect
from tests.mall_fixtures import detail, selection


class PaginationFlowTests(unittest.IsolatedAsyncioTestCase):
    async def flow(self, category, more, fail_next=False, intent='商品推荐'):
        params = FilterParams(category=category, max_price='300', color='黑色')
        model = AsyncMock(return_value=None)
        model.ainvoke.return_value = AIMessage(content=json.dumps({
            'intent':intent,'subjects':[],'needs_clarification':False}))
        cursor = 101 if category=='电子' else 'opaque-next'
        first = selection(detail(), has_more=more, cursor=str(cursor) if more else None)
        if category=='电子':
            first.page = ProductPage(items=first.page.items, size=3, hasMore=more, nextCursor=cursor if more else None)
        second = selection(detail(102))
        with tempfile.TemporaryDirectory() as directory:
            db = directory+'/pagination.db'
            with app(db) as (main, _, products), patch.object(main.router,'client',model), \
                    patch.object(mall,'list_categories',return_value=[CategoryNav(id=7,name='电子',level=1,hasChildren=False,sortOrder=0)]), \
                    patch('agent.nodes.recommend.invoke_model',new_callable=AsyncMock,return_value=params) as extract:
                await collect(main,'推荐电子商品' if intent=='商品推荐' else '统计电子商品','page')
                keyword_search = products.find_products.await_args.args[0]
                self.assertEqual(keyword_search.q, category)
                self.assertIsNone(keyword_search.categoryId)
                products.find_products.reset_mock()
                products.find_products.side_effect = [first, MallError('http',http_status=503), second] if fail_next else [first,second]
                await collect(main,'浏览分类','page')
                products.find_products.assert_not_awaited()
                await collect(main,'第一个','page')
                replies = [await collect(main,'下一批','page'), await collect(main,'下一批','page')]
                requests = [call.args[0] for call in products.find_products.await_args_list]
                self.assertEqual(len(requests), 3 if fail_next else 2 if more else 1)
                self.assertTrue(all(r.categoryId==7 and str(r.maxPrice)=='300' for r in requests))
                self.assertTrue(all(call.args[1].color=='黑色' for call in products.find_products.await_args_list))
                self.assertIsNone(requests[0].afterId if category=='电子' else requests[0].cursor)
                for req in requests[1:]:
                    self.assertEqual(req.afterId if category=='电子' else req.cursor, cursor)
                if not fail_next:
                    self.assertIn('没有下一页',replies[-1])
                model.ainvoke.assert_awaited_once()
                extract.assert_awaited_once()
            async with AsyncSqliteSaver.from_conn_string(db) as saver:
                values=(await main.builder.compile(checkpointer=saver).aget_state({'configurable':{'thread_id':'page'}})).values
            self.assertEqual(values['task_context']['intent'],'概览筛选')
            self.assertEqual(values['category_context']['selected']['id'],7)
            self.assertEqual(values['task_context']['filters']['color'],'黑色')
            self.assertEqual(values['category_action'],{})

    async def test_terminal_next_page_twice_keeps_range_without_requery(self):
        for category in ('电子','电子商品'):
            with self.subTest(category=category):
                await self.flow(category,False)

    async def test_next_page_preserves_cursor_and_then_stays_at_end(self):
        for category in ('电子','电子商品'):
            with self.subTest(category=category):
                await self.flow(category,True,intent='概览筛选')

    async def test_failed_page_can_retry_same_cursor_by_explicit_user_request(self):
        await self.flow('电子商品',True,True)

    async def test_missing_context_or_changed_conditions_still_use_router(self):
        router=IntentRouter()
        for query,task,page in [('下一批',{},{}),('下一批',{'intent':'业务咨询'}, {'hasMore':True}),
                                ('下一批，预算改为100',{'intent':'商品推荐'},{'hasMore':True})]:
            with self.subTest(query=query,task=task), patch.object(router,'decision',new_callable=AsyncMock,return_value=('澄清',[])) as decide:
                result=await router.node({'messages':[HumanMessage(content=query)],'task_context':task,'product_page':page})
                decide.assert_awaited_once()
                self.assertEqual(result['category_action'],{})
