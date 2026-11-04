"""概览只展示本批数量和分页状态，与推荐共用真实商品查询。"""
from agent.nodes.recommend import product_search_node
from agent.state import AgentState


async def total_node(state: AgentState):
    return await product_search_node(state, "概览筛选")
