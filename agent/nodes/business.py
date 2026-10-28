"""业务咨询节点：走 RAG，答案必须来自知识库。"""

from agent.deps import client, retriever
from agent.memory import AgentState, memory_manager
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.business")


async def business_node(state: AgentState):
    user_query = state["messages"][-1].content
    context = retriever.get_context(user_query)

    msgs = await memory_manager.build_bussiness_messages(state, user_query, context)
    logger.info(f"💬 business_node | 问题「{user_query[:30]}」| 上下文 {len(context)} 字符")

    response = await client.ainvoke(msgs)
    logger.info(f"✅ business_node 回答完成 | {len(response.content)} 字符")
    return {"messages": [response]}
