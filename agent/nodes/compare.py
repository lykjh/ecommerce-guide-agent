"""对比路由确认的对象；商城详情不齐时先让用户确认。"""
from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.budget import invoke_model
from agent.deps import get_client
from agent.prompts import COMPARE_ANSWER_SYSTEM, COMPARE_REQUEST, mall_error_reply
from agent.state import AgentState
from agent.tool.product import resolve_product
from agent.tool.schemas import CandidateRef, MallError


async def compare_shop_node(state: AgentState):
    candidates = [CandidateRef.model_construct(**item) for item in state.get("candidate_products", [])]
    details = []
    for subject in state["route_subjects"]:
        try:
            result = await resolve_product(subject, candidates)
        except ValidationError:
            reply = "商品名称过长或查询参数不合法，请提供要对比的商品编号。"
            return {"messages": [AIMessage(content=reply)], "final_reply": reply}
        except MallError as exc:
            reply = mall_error_reply(exc.kind, exc.http_status, exc.code)
            return {"messages": [AIMessage(content=reply)], "final_reply": reply}
        if result.status != "resolved":
            reply = f"尚未确认“{subject}”，请提供它的商品编号后再对比。"
            if result.status == "choose":
                reply += "\n" + "\n".join(f"- {item.name}（商品编号 {item.product_id}）" for item in result.candidates)
            return {"messages": [AIMessage(content=reply)], "final_reply": reply,
                    "candidate_products": [item.model_dump() for item in result.candidates]}
        details.append(result.product)
    names = [product.name for product in details]
    messages = [SystemMessage(content=COMPARE_ANSWER_SYSTEM.format(context="\n".join(p.model_dump_json() for p in details))),
                HumanMessage(content=COMPARE_REQUEST.format(names=", ".join(names))),
                HumanMessage(content=state["messages"][-1].content)]
    response = await invoke_model(get_client(), messages, purpose="answer")
    return {"messages": [response], "final_reply": response.content,
            "task_context": {"intent": "商品对比", "product_names": names},
            "candidate_products": [{"product_id": p.id, "name": p.name} for p in details]}
