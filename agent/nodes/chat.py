"""闲聊节点：不查知识库，只做礼貌应答并引导用户咨询商品。"""

from agent.deps import client
from agent.memory import AgentState, memory_manager
from utils.log_tool import get_loger

logger = get_loger("agent.nodes.chat")


async def chat_node(state: AgentState):
    # 一行代码搞定闲聊历史
    msgs = await memory_manager.build_chat_messages(state)

    response = await client.ainvoke(msgs)
    logger.info(f"✅ chat_node 回答完成 | {len(response.content)} 字符")
    return {"messages": [response]}
