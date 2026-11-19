"""Business evidence keeps tool data below instructions without dropping descriptions."""
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent.nodes import business
from agent.tool.schemas import ProductDetail, ProductResolution

class BusinessEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_queried_detail_is_tool_evidence_and_user_query_is_unchanged(self):
        fixtures = json.loads((Path(__file__).parent/'data/mall_products.json').read_text(encoding='utf8'))
        for key in ('101', '101-injected'):
            with self.subTest(fixture=key):
                detail = ProductDetail.model_validate(fixtures[key])
                query = HumanMessage(content='商品编号101多少钱？')
                resolve = AsyncMock(return_value=ProductResolution(status='resolved', product=detail))
                invoke = AsyncMock(return_value=AIMessage(content='测试回复'))
                with patch.object(business, 'resolve_product', resolve), patch.object(business, 'invoke_model', invoke), patch.object(business, 'get_client'):
                    result = await business.business_node({'messages':[query], 'route_subjects':['101']})
                sent = invoke.await_args.args[1]
                tool = sent[-2]
                self.assertIsInstance(tool, ToolMessage)
                self.assertEqual(json.loads(tool.content), json.loads(detail.model_dump_json()))
                self.assertEqual(sent[-3].tool_calls[0]['id'], tool.tool_call_id)
                self.assertEqual(sent[-3].tool_calls[0]['args'], {'product_id':101})
                self.assertEqual([m.content for m in sent if isinstance(m,HumanMessage)], [query.content, '仅回答本轮问题，不补答历史未完成或已取消的任务：\n' + query.content])
                self.assertNotIn(detail.description, sent[0].content)
                self.assertEqual(result['candidate_products'], [{'product_id':101,'name':detail.name}])
                resolve.assert_awaited_once_with('101', [])
                invoke.assert_awaited_once()
