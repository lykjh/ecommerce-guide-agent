"""Exercise the real structured-output adapter with a local HTTP transport; no API calls."""
import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent.deps import APP_CFG
from agent.nodes.recommend import recommend_node
from agent.tool.schemas import CategoryNav


class FilterTransportTests(unittest.IsolatedAsyncioTestCase):
    async def extract(self, values):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertNotIn('response_format', body)
            self.assertEqual(body['tool_choice']['function']['name'], 'FilterParams')
            schema = body['tools'][0]['function']['parameters']
            self.assertEqual(schema['properties']['unsupported_specs']['type'], 'array')
            return httpx.Response(200, json={
                'id': 'local', 'object': 'chat.completion', 'created': 0,
                'model': APP_CFG.get('llm.router_model'),
                'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': 'extract', 'type': 'function', 'function': {
                            'name': 'FilterParams', 'arguments': json.dumps(values)}}]}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
            client = ChatOpenAI(model=APP_CFG.get('llm.router_model'), api_key='local-test',
                                base_url='http://local.test/v1', http_async_client=transport,
                                max_retries=0)
            with patch('agent.nodes.recommend.get_extract_client', return_value=client), \
                    patch('agent.nodes.recommend.mall.list_categories', return_value=[
                        CategoryNav(id=1, name='电子', level=1, hasChildren=False, sortOrder=0)
                    ]) as categories:
                result = await recommend_node({'messages': [HumanMessage(content='推荐300元以内黑色电子商品')]})
        self.assertEqual(len(requests), 1)
        return result, categories

    async def test_real_adapter_carries_schema_and_parses_filters(self):
        result, categories = await self.extract({'category': '电子', 'max_price': '300',
                                                 'min_price': None, 'color': '黑色', 'next_page': False, 'unsupported_specs': []})
        self.assertEqual(result['task_context']['filters']['max_price'], '300')
        self.assertEqual(result['task_context']['filters']['color'], '黑色')
        self.assertTrue(result['category_context']['pending'])
        categories.assert_called_once()

    async def test_invalid_external_array_fails_without_query_or_default(self):
        result, categories = await self.extract({'category': '电子', 'unsupported_specs': {}})
        self.assertIn('筛选条件解析失败', result['final_reply'])
        self.assertNotIn('task_context', result)
        categories.assert_not_called()

    async def test_text_none_is_rejected_instead_of_silently_clearing_price(self):
        with self.assertLogs('agent.nodes.recommend', level='WARNING') as logs:
            result, categories = await self.extract({'category': '电子', 'min_price': 'None',
                                                     'unsupported_specs': []})
        self.assertIn('min_price', logs.output[0])
        self.assertIn('decimal_parsing', logs.output[0])
        self.assertNotIn("'None'", logs.output[0])
        self.assertIn('筛选条件解析失败', result['final_reply'])
        self.assertNotIn('task_context', result)
        categories.assert_not_called()

