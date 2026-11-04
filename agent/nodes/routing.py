"""固定边界回应，不执行未接入的业务或调用回答模型。"""

from langchain_core.messages import AIMessage

from agent.state import AgentState
from agent.prompts import ROUTE_REPLIES


async def route_reply_node(state: AgentState):
    reply = ROUTE_REPLIES.get(state.get("intent"), ROUTE_REPLIES["识别失败"])
    return {"messages": [AIMessage(content=reply)], "final_reply": reply}
