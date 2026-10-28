
from dotenv import load_dotenv

from tool.get_total_product import search_products_mock, count_products, MOCK_PRODUCTS

load_dotenv()

from utils.config_tool import get_app_config, get_rag_config
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk
import asyncio
import os
from agent.memory import AgentState, memory_manager, FilterParams,CompareParams
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rag.retriever import Retriever
retriever = Retriever()
from agent.router import router
from langgraph.graph import END
from utils.log_tool import get_loger
logger = get_loger("agent.main")

#配置统一从 config/app.yml 读取（原来写死在 mmm.py 里的常量）
APP_CFG = get_app_config()
#RAG 配置：距离阈值等（config/chroma.yml）
RAG_CFG = get_rag_config()
#SQLite 记忆库绝对路径
DB_PATH = APP_CFG.abs_path("path.db")
#模型服务地址。
#⚠️ 地址不写在 yml 里（专属网关带租户标识，仓库公开会泄露），
#只从环境变量读：yml 里配的是变量名 llm.base_url_env，真实值放 .env
BASE_URL = os.getenv(APP_CFG.get("llm.base_url_env"))

client =init_chat_model(
    model=APP_CFG.get("llm.model", "qwen-plus"),
    model_provider=APP_CFG.get("llm.model_provider", "openai"),
    api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY")),
    base_url=BASE_URL,
    temperature=APP_CFG.get("llm.temperature", 0.7),# 温度参数，控制生成文本的随机性
    max_tokens=APP_CFG.get("llm.max_tokens", 1024),# 最大生成 token 数，控制生成文本的长度
    #top_p=1,# 控制生成文本的多样性
    )

#参数抽取专用模型：抽取是「结构化提取」不是「自由创作」，必须用低温度 + 快模型。
#用上面那个 temperature=0.7 的 qwen-plus 做抽取，既慢又抽不稳（同样的话两次结果可能不同）。
extract_client = init_chat_model(
    model=APP_CFG.get("llm.router_model", "qwen-flash"),
    model_provider=APP_CFG.get("llm.model_provider", "openai"),
    api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY")),
    base_url=BASE_URL,
    temperature=0,
)

#所有「内部调用」统一打这个 tag，流式输出时靠它把抽取过程挡在界面外。
#（stream_mode="messages" 捕获的是节点内每一次 LLM 调用，不是每个节点一次）
INTERNAL_TAG = "internal"


async def business_node(state: AgentState):
    user_query=state["messages"][-1].content
    context=retriever.get_context(user_query)

    msgs=await memory_manager.build_bussiness_messages(state,user_query,context)
    logger.info(f"💬 business_node | 问题「{user_query[:30]}」| 上下文 {len(context)} 字符")

    response = await client.ainvoke(msgs)
    logger.info(f"✅ business_node 回答完成 | {len(response.content)} 字符")
    return {"messages":[response]}


async def chat_node(state: AgentState):
    # 一行代码搞定闲聊历史
    msgs = await memory_manager.build_chat_messages(state)

    response = await client.ainvoke(msgs)
    return {"messages": [response]}

async def total_node(state: AgentState):
    """概览筛选：抽取筛选条件 → 查库 → 让模型汇总。
    注意：抽取必须带上历史，否则「便宜点的呢」这种追问抽不出任何参数。
    """
    user_query=state["messages"][-1].content
    logger.info(f"🔎 total_node 进入 | 问题「{user_query[:30]}」")

    #打 internal tag：抽取过程属于内部调用，不能流到界面上
    structured_llm=(
        extract_client
        .with_structured_output(FilterParams)
        .with_config({"tags": [INTERNAL_TAG]})
    )

    try:
        #带上摘要和最近几轮历史，让追问（"便宜点的呢"/"有没有红色的"）也能抽到参数
        summary=state.get("summary", "")
        history=state["messages"][:-1][-6:]

        extract_msgs=[SystemMessage(content=(
            "你是电商筛选条件提取器。从用户输入中提取筛选条件，只提取用户明确提到的，"
            "没提到的留空。可结合历史对话补全代词和省略（如「便宜点的」= 比上文更低的预算）。"
        ))]
        if summary:
            extract_msgs.append(SystemMessage(content=f"【历史会话摘要】\n{summary}"))
        extract_msgs+=list(history)
        extract_msgs.append(HumanMessage(content=f"当前用户输入：{user_query}"))

        params=await structured_llm.ainvoke(extract_msgs)
        logger.info(f"✅ 筛选参数抽取完成 | {params.model_dump()}")
    except Exception as e:
        #不能静默吞掉：抽取失败和「用户没说条件」是两回事，排查时全靠这行日志
        logger.exception(f"❌ 筛选参数抽取失败，回退为空条件 | 问题「{user_query[:30]}」| {e}")
        params=FilterParams()

    if not any([params.category,params.max_price,params.min_price,params.color]):
        logger.warning(f"⚠️ 未抽到任何筛选条件，转为反问 | 问题「{user_query[:30]}」")
        return {"messages": [AIMessage(content="请问您想筛选什么品类、价格区间或颜色的商品呢？")]}

    #⚠️ 当前数据源是 tool/get_total_product.py 里的 MOCK_PRODUCTS（仅 4 条示例数据），
    #和向量库里那 2500 条真实商品不是同一份数据。等商品表建成后这里要换成真实查询。
    logger.warning("⚠️ 概览筛选当前走 MOCK_PRODUCTS 示例数据，尚未接入真实商品表")

    db_results=search_products_mock(
        category=params.category,
        max_price=params.max_price,
        min_price=params.min_price,
        color=params.color
    )
    #总数必须单独取：db_results 被截断到 10 条，直接 len() 会让模型以为一共就这么点
    total_count=count_products(
        category=params.category,
        max_price=params.max_price,
        min_price=params.min_price,
        color=params.color
    )
    logger.info(f"📦 查库完成 | 命中 {total_count} 条，本次送入模型 {len(db_results)} 条")

    if not db_results:
        logger.warning(f"⚠️ 无命中结果 | 条件 {params.model_dump()}")
        return {"messages": [AIMessage(content="很遗憾，没有找到符合您要求的商品。")]}

    context_str="\n".join(
        [f"- {p['name']} | 价格: {p['price']}元 | 颜色: {p['color']} | 类别: {p['category']}"
         for p in db_results]
    )

    #把「总数 vs 本次展示数」写进提示词，否则模型会把 10 条说成全部
    system_prompt=SystemMessage(content=(
        "你是导购助手。请根据下面的查库结果，向用户简要汇报概览。\n"
        "【绝对规则】\n"
        "1. 只说查库结果里有的商品，绝对不要编造任何商品、价格或参数。\n"
        "2. 必须如实告知总数和本次展示数量，不能让用户以为下面这些就是全部。\n"
        f"【查库结果】共命中 {total_count} 个商品，以下是前 {len(db_results)} 个：\n{context_str}"
    ))

    final_msgs=[system_prompt, HumanMessage(content=user_query)]
    response=await client.ainvoke(final_msgs)
    logger.info(f"✅ total_node 回答完成 | {len(response.content)} 字符")
    return {"messages": [response]}


async def fetch_product_data(product_name: str):
    """根据商品名称从向量库取详情。

    :return: 找到就返回内容字符串；**没找到返回 None**。
    返回 None 而不是「未找到」这种占位文本，是为了让调用方能明确区分「有数据」和「没数据」——
    把占位文本混进提示词，模型会拿着空信息硬编一份参数出来。
    """
    #对比场景用更严的阈值：宁可查不到，也不能召回无关商品冒充它
    threshold=RAG_CFG.get("compare_max_distance", 0.45)

    try:
        results=retriever.get_context(product_name, max_distance=threshold)
        if not results:
            logger.warning(f"⚠️ 未找到【{product_name}】的相关数据（阈值 {threshold}，可能库中确无此商品）")
            return None
        logger.info(f"✅ 获取【{product_name}】详情成功，{len(results)} 字符")
        return f"【{product_name}】的详情：\n{results}"
    except Exception as e:
        logger.exception(f"❌ 查询商品【{product_name}】异常: {e}")
        return None


async def compare_shop_node(state: AgentState):
    user_query=state["messages"][-1].content
    logger.info(f"⚖️ compare_shop_node 进入 | 问题「{user_query[:30]}」")

    structured_llm=(
        extract_client
        .with_structured_output(CompareParams)
        .with_config({"tags": [INTERNAL_TAG]})
    )

    try:
        extract_params=[
            SystemMessage(content="你是一个意图提取助手。请结合历史对话，提取出用户当前想要对比的商品名称。如果是代词，请去历史里找真实商品名。")
        ]+state["messages"]

        params=await structured_llm.ainvoke(extract_params)
        products_to_compare=params.product_names
        logger.info(f"✅ 对比商品名抽取完成 | {products_to_compare}")
    except Exception as e:
        #区分「抽取失败」和「商品查不到」，以前两者返回同一句话，线上根本没法定位
        logger.exception(f"❌ 对比商品名抽取失败 | 问题「{user_query[:30]}」| {e}")
        return {"messages": [AIMessage(content="没太明白您想对比哪两款商品，请具体说出它们的名称哦。")]}

    if len(products_to_compare) < 2:
        logger.warning(f"⚠️ 抽取到的商品不足 2 个 | {products_to_compare}")
        return {"messages": [AIMessage(content="请具体说出您想对比的两款商品的名称哦。")]}

    product_data_list=[]
    missing=[]
    for name in products_to_compare:
        data=await fetch_product_data(name)
        if data is None:
            missing.append(name)
        else:
            product_data_list.append(data)

    #🔴 关键：只要有一个商品没查到就到此为止。
    #把「未找到」喂给模型让它对比，模型只能凭空编参数——那还不如直接说没有。
    if missing:
        logger.warning(f"⚠️ 部分商品未查到，终止对比 | 缺失 {missing}")
        return {"messages": [AIMessage(
            content=f"抱歉，商品库里没有 {'、'.join(missing)} 的信息，没法对比。换个商品名试试？"
        )]}

    #join 必须在循环外：放在循环里每次都会覆盖，一旦将来加 continue 就会漏数据
    context_str="\n".join(product_data_list)
    logger.info(f"⚖️ 开始对比 | 商品 {products_to_compare} | 素材 {len(context_str)} 字符")

    system_prompt = SystemMessage(
        content=(
            "你是导购助手。请根据以下商品信息，向用户对比两款商品的异同。\n"
            "【绝对规则】只使用下面给出的信息，任何没提到的参数都不要编；"
            "如果某项参数资料里没有，就说「资料里没有提到」。\n"
            f"商品信息：\n{context_str}"
        )
    )

    compare_request=HumanMessage(content=f"请帮我对比：{', '.join(products_to_compare)}")
    response = await client.ainvoke([system_prompt, compare_request])
    return {"messages": [response]}


# async def recommend_node(state: AgentState):
#     # 推荐导购
#     response = await client.ainvoke(state["messages"])
#     return {"messages": [response]}
#
#
# async def unauthorized_node(state: AgentState):
#     # 越权拒答，不用结合记忆
#     return {"messages": [SystemMessage(content="你没有权限访问这个功能")]}


builder=StateGraph(AgentState)

#注册两个「干活」的节点：业务问答 / 闲聊
builder.add_node("business_node", business_node)
builder.add_node("chat_node", chat_node)
builder.add_node("summarize_node", memory_manager.summarize_node)
builder.add_node("total_node", total_node)
builder.add_node("compare_shop_node", compare_shop_node)
# builder.add_node("recommend_node", recommend_node)
# builder.add_node("unauthorized_node", unauthorized_node)

#⚠️ router 不能注册成普通节点：
#节点返回值必须是 dict（用来更新 state），而 router 返回的是 "闲聊"/"业务咨询" 这类字符串，
#注册成节点会直接报 InvalidUpdateError: Expected dict, got 闲聊。
#正确姿势是让它只当 START 的「分支函数」——只负责挑下一个节点，不产生 state 更新。
#这样每轮也只调用一次路由模型（注册成节点会被调用两次：进节点一次、判分支一次）。
#意图 -> 节点 的映射表。新增意图要同时改三处：这里、router 的 RouteName、router 的 prompt。
ROUTE_MAP = {
    "闲聊": "chat_node",
    "业务咨询": "business_node",
    "概览筛选": "total_node",
    "商品对比": "compare_shop_node",
}

#会产出「给用户看的内容」的节点，流式输出只放行这些
VISIBLE_NODES = ("business_node", "chat_node", "total_node", "compare_shop_node")


async def route_with_fallback(state) -> str:
    """在 router 外面包一层兜底。

    🔴 不加这层会直接崩：router 的 prompt 里列了 7 类意图，但只要返回的意图不在
    ROUTE_MAP 里，LangGraph 会抛 `KeyError: '商品推荐'`，整轮对话直接挂掉。
    兜底策略是记日志 + 转业务咨询（RAG 至少能答），而不是默默吞掉——
    日志里会留下「哪个意图没接」，方便你后面补实现。
    """
    intent = await router.router(state)
    if intent not in ROUTE_MAP:
        logger.warning(
            f"⚠️ 意图「{intent}」尚未接入实现，兜底到业务咨询。"
            f"已实现: {sorted(ROUTE_MAP)}"
        )
        return "业务咨询"
    logger.info(f"🧭 路由结果: {intent} -> {ROUTE_MAP[intent]}")
    return intent


builder.add_conditional_edges(
    START,#从入口直接分支，不经过多余的 router 节点
    route_with_fallback,
    ROUTE_MAP,
)
#概览筛选也要进摘要节点：它同样是一轮真实对话，不进的话这轮就丢了
builder.add_edge("total_node", "summarize_node")
builder.add_edge("compare_shop_node","summarize_node")
builder.add_edge("business_node", "summarize_node")
builder.add_edge("chat_node", "summarize_node")
builder.add_edge("summarize_node", END)
#本地。db文件存储作为记忆系统

async def get_ai_stream(user_input: str,thread_id:str="default"):
    config={"configurable":{"thread_id":thread_id}}
    logger.info(f"▶️ 新一轮对话 | thread={thread_id} | 输入「{user_input[:40]}」")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph=builder.compile(checkpointer=checkpointer)

        emitted=False

        async for msg_chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
        ):

            #1) 先挡掉「内部调用」：参数抽取这类中间过程不能让用户看到。
            #   stream_mode="messages" 捕获的是节点内每一次 LLM 调用，不是每个节点一次，
            #   所以只按节点名过滤是不够的，必须按调用粒度挡。
            if INTERNAL_TAG in (metadata.get("tags") or []):
                continue

            #2) 再按节点过滤：路由模型自己输出的意图标签归属 __start__，也要挡掉
            node_name=metadata.get("langgraph_node")
            if (
                msg_chunk.type == "AIMessageChunk"
                and msg_chunk.content
                and node_name in VISIBLE_NODES
            ):
                emitted=True
                yield msg_chunk.content

        #3) 兜底：节点直接 return AIMessage(...) 的「写死回复」（例如"没找到该商品"）
        #   不经过 LLM，stream_mode="messages" 不会产出任何 chunk，用户会看到一片空白。
        #   这种情况从最终状态里把最后一条 AI 回复取出来一次性吐出去。
        if not emitted:
            logger.warning("⚠️ 本轮无流式输出（节点返回的是预设回复），改为从状态里取最终消息")
            try:
                final_state=await graph.aget_state(config)
                for msg in reversed(final_state.values.get("messages", [])):
                    #排除 AIMessageChunk：那是流式的碎片，不是完整回复
                    if isinstance(msg, AIMessage) and not isinstance(msg, AIMessageChunk) and msg.content:
                        yield msg.content
                        break
            except Exception as e:
                logger.exception(f"❌ 取最终消息失败: {e}")


if __name__ == "__main__":
    async def test():
        print("测试开始：")
        async for chunk in get_ai_stream("无线蓝牙耳机多少钱"):
            print(chunk, end="", flush=True)
        print("\n")

        # 测一下记忆：换个问法问上面那句话
        async for chunk in get_ai_stream("智能保温杯还有货吗"):
            print(chunk, end="", flush=True)
        print("\n")

    asyncio.run(test())


