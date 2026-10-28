"""商品对比节点：抽出要对比的商品名 → 逐个取详情 → 让模型对比。

🔴 这个节点最要紧的一条规则：**只要有任何一个商品没查到，就到此为止**。
把「未找到」拼进提示词让模型对比，模型只能凭空编参数——那还不如直接说没有。
（修这个之前，截图里的对比回答就是这么编出来的）
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.deps import client, extract_client, INTERNAL_TAG
from agent.memory import AgentState, CompareParams
from agent.prompts import (
    COMPARE_ANSWER_SYSTEM,
    COMPARE_EXTRACT_SYSTEM,
    COMPARE_NEED_TWO,
    COMPARE_NOT_UNDERSTOOD,
    COMPARE_REQUEST,
    compare_missing,
)
from tool.product import fetch_product_data
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.compare")


async def compare_shop_node(state: AgentState):
    user_query = state["messages"][-1].content
    logger.info(f"⚖️ compare_shop_node 进入 | 问题「{user_query[:30]}」")

    structured_llm = (
        extract_client
        .with_structured_output(CompareParams)
        .with_config({"tags": [INTERNAL_TAG]})
    )

    try:
        extract_params = [SystemMessage(content=COMPARE_EXTRACT_SYSTEM)] + state["messages"]

        params = await structured_llm.ainvoke(extract_params)
        products_to_compare = params.product_names
        logger.info(f"✅ 对比商品名抽取完成 | {products_to_compare}")
    except Exception as e:
        #区分「抽取失败」和「商品查不到」，以前两者返回同一句话，线上根本没法定位
        logger.exception(f"❌ 对比商品名抽取失败 | 问题「{user_query[:30]}」| {e}")
        return {"messages": [AIMessage(content=COMPARE_NOT_UNDERSTOOD)]}

    if len(products_to_compare) < 2:
        logger.warning(f"⚠️ 抽取到的商品不足 2 个 | {products_to_compare}")
        return {"messages": [AIMessage(content=COMPARE_NEED_TWO)]}

    product_data_list = []
    missing = []
    for name in products_to_compare:
        data = await fetch_product_data(name)
        if data is None:
            missing.append(name)
        else:
            product_data_list.append(data)

    if missing:
        logger.warning(f"⚠️ 部分商品未查到，终止对比 | 缺失 {missing}")
        return {"messages": [AIMessage(content=compare_missing(missing))]}

    #join 必须在循环外：放在循环里每次都会覆盖，一旦将来加 continue 就会漏数据
    context_str = "\n".join(product_data_list)
    logger.info(f"⚖️ 开始对比 | 商品 {products_to_compare} | 素材 {len(context_str)} 字符")

    system_prompt = SystemMessage(content=COMPARE_ANSWER_SYSTEM.format(context=context_str))
    compare_request = HumanMessage(content=COMPARE_REQUEST.format(names=", ".join(products_to_compare)))

    response = await client.ainvoke([system_prompt, compare_request])
    logger.info(f"✅ compare_shop_node 回答完成 | {len(response.content)} 字符")
    return {"messages": [response]}
