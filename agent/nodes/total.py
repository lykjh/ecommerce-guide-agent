"""概览筛选节点：抽取筛选条件 → 查库 → 让模型汇总。

两条容易踩的规则写在这里，别在别处重复实现：
1. 抽取**必须带历史**，否则「便宜点的呢」这种追问抽不出参数。
2. 总数要用 count_products() 单独取，不能用 len(搜索结果)——
   搜索结果被截断过，直接拿长度会让模型告诉用户「一共就这些」。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.deps import client, extract_client, INTERNAL_TAG
from agent.memory import AgentState, FilterParams
from agent.prompts import (
    ASK_FILTER_CONDITION,
    NO_PRODUCT_MATCHED,
    TOTAL_ANSWER_SYSTEM,
    TOTAL_EXTRACT_SYSTEM,
    TOTAL_EXTRACT_USER,
)
from tool.product import count_products, search_products_mock
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.total")

#最多回看几轮历史来补全省略
HISTORY_TURNS = 6


async def total_node(state: AgentState):
    user_query = state["messages"][-1].content
    logger.info(f"🔎 total_node 进入 | 问题「{user_query[:30]}」")

    #打 internal tag：抽取过程属于内部调用，不能流到界面上
    structured_llm = (
        extract_client
        .with_structured_output(FilterParams)
        .with_config({"tags": [INTERNAL_TAG]})
    )

    try:
        #带上摘要和最近几轮历史，让追问（"便宜点的呢"/"有没有红色的"）也能抽到参数
        summary = state.get("summary", "")
        history = state["messages"][:-1][-HISTORY_TURNS:]

        extract_msgs = [SystemMessage(content=TOTAL_EXTRACT_SYSTEM)]
        if summary:
            extract_msgs.append(SystemMessage(content=f"【历史会话摘要】\n{summary}"))
        extract_msgs += list(history)
        extract_msgs.append(HumanMessage(content=TOTAL_EXTRACT_USER.format(user_query=user_query)))

        params = await structured_llm.ainvoke(extract_msgs)
        logger.info(f"✅ 筛选参数抽取完成 | {params.model_dump()}")
    except Exception as e:
        #不能静默吞掉：抽取失败和「用户没说条件」是两回事，排查时全靠这行日志
        logger.exception(f"❌ 筛选参数抽取失败，回退为空条件 | 问题「{user_query[:30]}」| {e}")
        params = FilterParams()

    if not any([params.category, params.max_price, params.min_price, params.color]):
        logger.warning(f"⚠️ 未抽到任何筛选条件，转为反问 | 问题「{user_query[:30]}」")
        return {"messages": [AIMessage(content=ASK_FILTER_CONDITION)]}

    #⚠️ 当前数据源是 tool/product.py 的 MOCK_PRODUCTS（仅 4 条示例数据），
    #和向量库里那 2500 条真实商品不是同一份数据。等商品表建成后这里换成真实查询。
    logger.warning("⚠️ 概览筛选当前走 MOCK_PRODUCTS 示例数据，尚未接入真实商品表")

    db_results = search_products_mock(
        category=params.category,
        max_price=params.max_price,
        min_price=params.min_price,
        color=params.color,
    )
    #总数必须单独取：db_results 被截断，直接 len() 会让模型以为一共就这么点
    total_count = count_products(
        category=params.category,
        max_price=params.max_price,
        min_price=params.min_price,
        color=params.color,
    )
    logger.info(f"📦 查库完成 | 命中 {total_count} 条，本次送入模型 {len(db_results)} 条")

    if not db_results:
        logger.warning(f"⚠️ 无命中结果 | 条件 {params.model_dump()}")
        return {"messages": [AIMessage(content=NO_PRODUCT_MATCHED)]}

    context_str = "\n".join(
        [f"- {p['name']} | 价格: {p['price']}元 | 颜色: {p['color']} | 类别: {p['category']}"
         for p in db_results]
    )

    #把「总数 vs 本次展示数」写进提示词，否则模型会把截断结果说成全部
    system_prompt = SystemMessage(content=TOTAL_ANSWER_SYSTEM.format(
        total=total_count,
        shown=len(db_results),
        context=context_str,
    ))

    final_msgs = [system_prompt, HumanMessage(content=user_query)]
    response = await client.ainvoke(final_msgs)
    logger.info(f"✅ total_node 回答完成 | {len(response.content)} 字符")
    return {"messages": [response]}
