"""商品数据访问层：所有「取商品」的动作都在这里。

目前有两条取数路径，各自用途不同：
1. `search_products_mock` / `count_products`：结构化筛选，查 MOCK_PRODUCTS。
   ⚠️ 这只是 4 条示例数据，和向量库里那 2500 条商品不是同一份数据。
   等真实商品表建成，这两个函数换成查表即可，上层节点不用改。
2. `fetch_product_data`：按商品名到向量库取详情片段。
   商品对比走这条，因为它要的是「一段描述」而不是结构化字段。

关键约定：`fetch_product_data` 查不到时返回 **None**，不返回「未找到」这种占位文本。
占位文本混进提示词，模型会拿着空信息硬编一份参数出来。
"""

from pydantic import BaseModel, Field

from rag.retriever import get_retriever
from utils.config_tool import get_rag_config
from utils.log_tool import get_loger

logger = get_loger("tool.product")

RAG_CFG = get_rag_config()


# ============================================================
# 一、结构化筛选（当前为示例数据）
# ============================================================

# 模拟你的商品数据库
MOCK_PRODUCTS = [
    {"id": 1, "name": "夏季纯棉T恤", "category": "衣服", "price": 99, "color": "白色"},
    {"id": 2, "name": "轻薄防晒衣", "category": "衣服", "price": 159, "color": "红色"},
    {"id": 3, "name": "iPhone 15", "category": "手机", "price": 4999, "color": "黑色"},
    {"id": 4, "name": "红米 K70", "category": "手机", "price": 1999, "color": "蓝色"},
]


class ProductSearchSchema(BaseModel):
    """给 LangChain @tool 用的参数结构（目前 @tool 未启用，接真实工具时直接用）"""
    category: str = Field(description="商品品类，如衣服、手机")
    min_price: float = Field(default=0, description="最低价格")
    max_price: float = Field(default=999999, description="最高价格")


def _filter_products(category=None, min_price=None, max_price=None, color=None):
    """纯筛选逻辑，不截断。count 和 search 共用，避免两处过滤规则写歪。"""
    results = MOCK_PRODUCTS

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


# ============================================================
# 二、向量库取详情
# ============================================================

async def fetch_product_data(product_name: str):
    """按商品名从向量库取详情。

    :return: 找到返回内容字符串；**没找到返回 None**。
    """
    #对比场景用更严的阈值：宁可查不到，也不能召回无关商品冒充它
    threshold = RAG_CFG.get("compare_max_distance", 0.45)

    try:
        results = get_retriever().get_context(product_name, max_distance=threshold)
        if not results:
            logger.warning(f"⚠️ 未找到【{product_name}】的相关数据（阈值 {threshold}，可能库中确无此商品）")
            return None
        logger.info(f"✅ 获取【{product_name}】详情成功，{len(results)} 字符")
        return f"【{product_name}】的详情：\n{results}"
    except Exception as e:
        logger.exception(f"❌ 查询商品【{product_name}】异常: {e}")
        return None
