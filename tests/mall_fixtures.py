"""人工商城数据，只用于离线测试。"""
from agent.tool.schemas import ProductDetail, ProductSelection, DiscoveryPage, ProductMatch

def detail(product_id=101, name="耳机甲", price="299.00"):
    return ProductDetail(id=product_id,name=name,description="续航8小时",options=[{"id":1,"name":"颜色","values":[{"id":11,"value":"黑色"}]}],skus=[{"id":product_id*10,"optionValueIds":[11],"price":price,"available":True}])

def selection(*products, has_more=False, cursor=None):
    return ProductSelection(matches=[ProductMatch(product=p,skus=p.skus) for p in products],
        page=DiscoveryPage(items=[{"id":p.id,"name":p.name,"startingPrice":p.skus[0].price,"soldOut":False} for p in products],hasMore=has_more,nextCursor=cursor),
        checked_product_ids=[p.id for p in products],unverified_product_ids=[])
