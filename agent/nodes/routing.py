"""政策检索与固定边界回应，不执行业务写操作。"""

from langchain_core.messages import AIMessage

from agent.state import AgentState
from agent.prompts import ROUTE_REPLIES
from agent.nodes.policy import policy_node


async def route_reply_node(state: AgentState):
    if state.get("intent") == "政策问答":
        return await policy_node(state)
    reply = ROUTE_REPLIES.get(state.get("intent"), ROUTE_REPLIES["识别失败"])
    return {"messages": [AIMessage(content=reply)], "final_reply": reply}
