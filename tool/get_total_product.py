from pydantic import  BaseModel,Field
from  langchain_core.tools import tool


# 模拟你的商品数据库
MOCK_PRODUCTS = [
    {"id": 1, "name": "夏季纯棉T恤", "category": "衣服", "price": 99, "color": "白色"},
    {"id": 2, "name": "轻薄防晒衣", "category": "衣服", "price": 159, "color": "红色"},
    {"id": 3, "name": "iPhone 15", "category": "手机", "price": 4999, "color": "黑色"},
    {"id": 4, "name": "红米K70", "category": "手机", "price": 1999, "color": "蓝色"},
]

class ProductSearchSchema(BaseModel):
    category: str = Field(description="商品品类，如衣服、手机")
    min_price: float = Field(default=0, description="最低价格")
    max_price: float = Field(default=999999, description="最高价格")

# @tool("search_products",args_schema=ProductSearchSchema)
# def search_products(category: str, min_price: float, max_price: float):
#     """根据条件筛选商品"""
#     return f"搜索 {category} 类别，价格在 {min_price} 到 {max_price} 之间的商品"


def _filter_products(category=None, min_price=None, max_price=None, color=None):
    """纯筛选逻辑，不截断。count 和 search 共用，避免两处过滤规则写歪。"""
    results=MOCK_PRODUCTS

    if category:
        results = [p for p in results if category in p["category"]]
    if min_price is not None:
        results = [p for p in results if p["price"] >= min_price]
    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]
    if color:
        results = [p for p in results if color in p["color"]]

    return results


def count_products(category=None, min_price=None, max_price=None, color=None) -> int:
    """符合条件的商品总数（不受 limit 影响）。

    必须和搜索分开：搜索结果被截断到 limit 条，直接用 len(搜索结果) 当总数，
    模型会告诉用户「一共就这些」，那是假数据。
    """
    return len(_filter_products(category, min_price, max_price, color))


def search_products_mock(category=None, min_price=None, max_price=None, color=None, limit=10):
    """按条件筛选商品，最多返回 limit 条。"""
    return _filter_products(category, min_price, max_price, color)[:limit]

