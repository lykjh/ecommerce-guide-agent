"""图状态；持久化由 checkpointer 负责。"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    signal_state: dict
    final_reply: str
    intent: str
    route_subjects: list[str]
    task_context: dict
    candidate_products: list[dict]
    comparison_products: list[dict]
    product_page: dict
    category_context: dict
    category_action: dict


