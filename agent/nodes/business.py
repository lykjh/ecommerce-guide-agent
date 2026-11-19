"""商品咨询：先确认商城商品，再解释本次详情。"""
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from agent.budget import invoke_model
from agent.deps import get_client
from agent.memory import memory_manager
from agent.prompts import BUSINESS_SYSTEM, mall_error_reply
from agent.state import AgentState
from agent.tool.product import resolve_product
from agent.tool.schemas import CandidateRef, MallError


async def business_node(state: AgentState):
    candidates = [CandidateRef.model_construct(**item) for item in state.get("candidate_products", [])]
    try:
        result = await resolve_product(state["route_subjects"][0], candidates)
    except ValidationError:
        reply = "商品名称过长或查询参数不合法，请提供商品编号或较短的完整名称。"
        return {"messages": [AIMessage(content=reply)], "final_reply": reply}
    except MallError as exc:
        reply = mall_error_reply(exc.kind, exc.http_status, exc.code)
        return {"messages": [AIMessage(content=reply)], "final_reply": reply}
    if result.status != "resolved":
        reply = "未查到该商品，请核对商品编号或完整名称；本次无法确认其价格和可售状态。"
        if result.status == "choose":
            reply = "请确认具体商品，回复商品编号：\n" + "\n".join(
                f"- {item.name}（商品编号 {item.product_id}）" for item in result.candidates)
        return {"messages": [AIMessage(content=reply)], "final_reply": reply,
                "candidate_products": [item.model_dump() for item in result.candidates]}
    product = result.product
    messages = memory_manager.build_messages(state, BUSINESS_SYSTEM)
    # 表达刚完成的真实查询，避免把商城自由文本提升为用户指令。
    messages.extend([
        AIMessage(content="", tool_calls=[{"name": "get_product_detail",
                  "args": {"product_id": product.id}, "id": "business_product_detail", "type": "tool_call"}]),
        ToolMessage(content=product.model_dump_json(), tool_call_id="business_product_detail",
                    name="get_product_detail"),
        HumanMessage(content="仅回答本轮问题，不补答历史未完成或已取消的任务：\n" + state["messages"][-1].content),
    ])
    response = await invoke_model(get_client(), messages, purpose="answer")
    return {"messages": [response], "final_reply": response.content,
            "task_context": {"intent": "业务咨询", "product_names": [product.name]},
            "candidate_products": [{"product_id": product.id, "name": product.name}]}
