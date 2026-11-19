"""Focused repair checks: controlled tools, no production writes or real-account claim."""
import asyncio,json,sys,logging
from pathlib import Path
from unittest.mock import AsyncMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from langchain_core.messages import HumanMessage,AIMessage
from agent.tool.schemas import FilterParams,DiscoveryRequest,ProductDetail
from agent.tool.product import find_products
from agent.nodes.recommend import product_search_node
from agent.memory import memory_manager

OUT=Path(sys.argv[1]);rows=[]
async def run():
    details={n:ProductDetail(id=n,name=f'商品{n}',description='',options=[],skus=[dict(id=n+100,price=p,available=True,optionValueIds=[])]) for n,p in [(1,'89.90'),(2,'59.90'),(3,'39.90')]}
    def detail(n):return details[n]
    with patch('agent.tool.product.mall.get_product',side_effect=detail) as get,patch('agent.tool.product.mall.search_products',new_callable=AsyncMock) as search,patch('agent.tool.product.mall.list_products',new_callable=AsyncMock) as listing:
        for maximum,expected in [('60',[2]),('100',[1,2]),('10',[])]:
            result=await find_products(DiscoveryRequest(),FilterParams(max_price=maximum),product_ids=[1,2])
            assert [m.product.id for m in result.matches]==expected
            assert result.checked_product_ids==[1,2] and not result.page.hasMore
            rows.append({'case':'fixed_scope_'+maximum,'passed':True})
        assert not search.called and not listing.called
    state={'messages':[HumanMessage(content='只在刚才两款中筛，预算60')],'comparison_products':[{'product_id':1,'name':'商品1'},{'product_id':2,'name':'商品2'}],'candidate_products':[{'product_id':3,'name':'商品3'}],'task_context':{}}
    with patch('agent.nodes.recommend.invoke_model',new_callable=AsyncMock,return_value=FilterParams(scope='comparison',max_price='60')),patch('agent.nodes.recommend.find_products',new_callable=AsyncMock,return_value=result) as find,patch('agent.nodes.recommend.navigate_categories',new_callable=AsyncMock) as nav:
        await product_search_node(state,'概览筛选');assert find.call_args.kwargs['product_ids']==[1,2] and not nav.called
        rows.append({'case':'scope_node_uses_saved_comparison','passed':True})
        state['comparison_products']=[];find.reset_mock();reply=await product_search_node(state,'概览筛选');assert not find.called and '未扩大' in reply['final_reply']
        rows.append({'case':'missing_scope_fails_closed','passed':True})
    for price in ['100.0','100.00','1E+2']:
        with patch('agent.nodes.recommend.invoke_model',new_callable=AsyncMock,return_value=FilterParams(category='海鲜',max_price=price)),patch('agent.nodes.recommend.find_products',new_callable=AsyncMock,return_value=result) as find:
            first=await product_search_node({'messages':[HumanMessage(content='海鲜100元以内')]},'概览筛选')
            initial=find.call_args.args[0].model_dump(mode='json')['maxPrice']
            await product_search_node({'messages':[HumanMessage(content='下一批')],'task_context':first['task_context'],'category_context':first['category_context'],'category_action':{'kind':'next_page'}},'概览筛选')
            resumed=find.call_args.args[0].model_dump(mode='json')['maxPrice']
            assert initial==resumed=='100'
            rows.append({'case':'price_cursor_serialization_'+price,'passed':True})
    state['comparison_products']=[{'product_id':1,'name':'商品1'},{'product_id':2,'name':'商品2'}]
    for purpose in ['route','extract']:
        text=''.join(m.content for m in memory_manager.build_messages(state,'test',purpose))
        assert 'comparison_products' in text and '商品1' in text and '商品2' in text
        rows.append({'case':'comparison_survives_'+purpose,'passed':True})
    from agent.nodes.policy import policy_node
    from agent.rag.policy import load_policy,retrieve
    query='我买了衣服不喜欢想退，运费谁出，几天内能申请，需要保持什么状态？'
    manifest,clauses=load_policy()
    assert '6.1' not in {c.id for c in retrieve(clauses,query)}
    with patch('agent.nodes.policy.invoke_model',new_callable=AsyncMock,side_effect=[AIMessage(content='{"clause_ids":["6.1","6.2","9"]}'),AIMessage(content='{"sentence_ids":["6.1#1","6.2#1","9#1"]}')]) as model:
        result=await policy_node({'messages':[HumanMessage(content=query)]})
        assert '签收次日起 7个自然日内' in result['final_reply'] and '原品质' in result['final_reply']
        catalog=json.loads(model.call_args_list[0].args[1][2].content)['clause_catalog']
        assert {'6.1','6.2','9'} <= {c['id'] for c in catalog}
        rows.append({'case':'missing_lexical_evidence_recovered_from_catalog','passed':True})
    (OUT/'focused-unit-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(rows)}/{len(rows)} passed')
asyncio.run(run())
