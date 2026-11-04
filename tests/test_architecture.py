from tests.mall_fixtures import detail, selection
from agent.tool.schemas import ProductResolution
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

from langchain_core.messages import AIMessage, HumanMessage
from agent.budget import invoke_query, TurnBudget, BudgetStop, budget_scope
from agent.tool.schemas import FilterParams
from tests.check_runtime_boundaries import app, collect


class ArchitectureTests(unittest.IsolatedAsyncioTestCase):
    def test_import_does_not_construct_external_clients(self):
        script = """
from unittest.mock import patch
with patch('langchain.chat_models.init_chat_model') as model, patch('langchain_chroma.Chroma') as db:
    import agent.main_agent
    model.assert_not_called()
    db.assert_not_called()
"""
        result = subprocess.run([sys.executable, '-B', '-c', script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    async def test_async_query_stays_on_caller_loop(self):
        loop = asyncio.get_running_loop()
        async def query():
            self.assertIs(asyncio.get_running_loop(), loop)
            return "ok"
        self.assertEqual(await invoke_query(query), "ok")

    async def test_async_timeout_cancels_and_releases_slot(self):
        closed = asyncio.Event()
        slot = threading.BoundedSemaphore(1)
        async def query():
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        with patch('agent.budget._QUERY_SLOTS', slot), patch.object(TurnBudget, 'QUERY_SECONDS', 0.01):
            with budget_scope(TurnBudget('async-timeout')):
                with self.assertRaisesRegex(BudgetStop, 'query_timeout'):
                    await invoke_query(query)
            self.assertTrue(closed.is_set())
            self.assertTrue(slot.acquire(blocking=False))
            slot.release()

    async def test_async_cancellation_and_failure_release_slot(self):
        slot = threading.BoundedSemaphore(1)
        entered, closed = asyncio.Event(), asyncio.Event()
        async def query():
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
        async def failed():
            raise RuntimeError('upstream failure')
        with patch('agent.budget._QUERY_SLOTS', slot):
            task = asyncio.create_task(invoke_query(query))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(closed.is_set())
            with self.assertRaisesRegex(RuntimeError, 'upstream failure'):
                await invoke_query(failed)
            self.assertTrue(slot.acquire(blocking=False))
            slot.release()

    async def test_comparison_uses_confirmed_subjects_once(self):
        from agent.nodes.compare import compare_shop_node
        state = {'messages': [HumanMessage(content='比较这两款')], 'route_subjects': ['耳机甲', '耳机乙']}
        with app(':memory:') as (_, deps, products):
            products.resolve_product.side_effect = [ProductResolution(status='resolved',product=detail()), ProductResolution(status='resolved',product=detail(102,'耳机乙'))]
            deps.client.ainvoke = AsyncMock(return_value=AIMessage(content='对比结果'))
            result = await compare_shop_node(state)
            self.assertEqual(products.resolve_product.call_args_list, [call('耳机甲', []), call('耳机乙', [])])
            deps.extract_client.with_structured_output.assert_not_called()
            deps.client.ainvoke.assert_awaited_once()
            self.assertEqual(result['task_context']['product_names'], state['route_subjects'])

    async def test_comparison_missing_evidence_does_not_generate(self):
        from agent.nodes.compare import compare_shop_node
        with app(':memory:') as (_, deps, products):
            products.resolve_product.side_effect = [ProductResolution(status='resolved',product=detail()), ProductResolution(status='not_found')]
            result = await compare_shop_node({'messages': [HumanMessage(content='比较两款')],
                                              'route_subjects': ['耳机甲', '耳机乙']})
            deps.client.ainvoke.assert_not_called()
            self.assertIn('耳机乙', result['final_reply'])

    async def test_product_search_preserves_decimal_and_propagates_failure(self):
        from agent.tool.product import find_products
        from agent.tool.schemas import DiscoveryRequest, MallError
        from decimal import Decimal
        request = DiscoveryRequest(q="耳机", maxPrice="0.10")
        with patch("agent.tool.product.mall.search_products", side_effect=MallError("http", http_status=503)) as search:
            with self.assertRaises(MallError):
                await find_products(request, FilterParams(category="耳机",max_price="0.10"))
        self.assertEqual(search.call_args.args[0].maxPrice, Decimal("0.10"))

    async def test_registered_recommendation_emits_final_reply(self):
        from agent.router import ROUTE_MAP
        with app(':memory:') as (main, _, _):
            self.assertTrue(set(ROUTE_MAP.values()).issubset(main.VISIBLE_NODES))
            router = AsyncMock()
            router.ainvoke.return_value = AIMessage(content=json.dumps({
                'intent': '商品推荐', 'subjects': [], 'needs_clarification': False}))
            with patch.object(main.router, 'client', router), \
                 patch('agent.nodes.recommend.invoke_model', new_callable=AsyncMock,
                       return_value=FilterParams(category='耳机')), \
                 patch('agent.nodes.recommend.find_products', new_callable=AsyncMock, return_value=selection(detail())):
                self.assertIn('商品编号 101', await collect(main, '推荐耳机', 'architecture-output'))
