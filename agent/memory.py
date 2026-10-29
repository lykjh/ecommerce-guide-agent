"""
内存模块，用于存储和检索对话历史。

"""

import os

from typing import List, Annotated, Optional
from pydantic import BaseModel, Field
from pydantic import BaseModel
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from agent.prompts import RECOMMEND_SYSTEM
from utils.config_tool import get_app_config

APP_CFG = get_app_config()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    # 闲聊中累积的购买意向信号，形如 {"signal": "购买意向", "level": 6}
    # 靠 checkpointer 随 thread_id 跨轮留存；达到 SIGNAL_THRESHOLD 时触发推荐，
    # 推荐后由 recommend_node 归零，避免用户每说一句闲聊就被推一次。
    signal_state: dict

class FilterParams(BaseModel):
    category: Optional[str] = Field(default=None, description="商品品类，如：手机、衣服、电脑。如果没有提到，填None")
    min_price: Optional[float] = Field(default=None, description="最低价格。如果没有提到，填None")
    max_price: Optional[float] = Field(default=None, description="最高价格。如果没有提到，填None")
    color: Optional[str] = Field(default=None, description="颜色，如：红色、黑色。如果没有提到，填None")

# agent/memory.py 里追加
from pydantic import BaseModel, Field
from typing import List

class CompareParams(BaseModel):
    product_names: List[str] = Field(
        description="用户想要对比的商品名称列表。如果用户使用了'这个'、'刚才那个'等代词，请务必结合对话历史替换为真实的商品名称。"
    )

class MemoryManager:
    def __init__(self):
        self.pronouns =["它", "这个", "那个", "这款", "那款", "刚才"]
        self.keywords=["耳机", "猫粮", "保温杯", "手机", "元", "发货", "库存"]
        self.summarizer_client = init_chat_model(
            model=APP_CFG.get("llm.router_model", "qwen-flash"),
            model_provider="openai",
            api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY")),
            base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
            temperature=0,
        )

    def _need_history(self, query: str) -> bool:
        return any(p in query for p in self.pronouns)

    def _has_keyword(self, content: str) -> bool:
        return any(kw in content for kw in self.keywords)

    def get_working_memory(self, messages: List[BaseMessage], current_query: str) -> List[BaseMessage]:
        """L1 短期阻断记忆"""
        if not self._need_history(current_query):
            return []

        recent_messages = list(messages[-20:])
        clean_history = []

        for i in range(len(recent_messages) - 2, -1, -1):
            msg = recent_messages[i]
            if isinstance(msg, SystemMessage):
                continue
            if self._has_keyword(msg.content):
                start_idx = max(0, i - 2)
                clean_history = recent_messages[start_idx:i + 2]
                break
        return clean_history

    async def build_bussiness_messages(self, state: AgentState, user_query: str, context: str)->List[BaseMessage]:
        current_summary =state.get("summary", "无")

        if not context:
            system_prompt = "你是一个专业的电商客服。请直接回答：“亲，暂时没有查到相关信息哦，建议您咨询人工客服~”"
        else:
            system_prompt =(
                f"你是一个专业的电商客服。\n"
                f"【历史会话摘要】\n{current_summary}\n\n"
                "请严格根据下面的【参考信息】回答用户问题。绝对不要自己编造。\n"
                f"【参考信息】\n{context}"
            )

        working_memory=self.get_working_memory(state["messages"],user_query)

        if not working_memory:
            working_memory=state["messages"][-4:-1]

        return [SystemMessage(content=system_prompt)]+working_memory+[state["messages"][-1]]

    async def build_chat_messages(self, state: AgentState, system_prompt: str)->List[BaseMessage]:
        """闲聊消息组装。

        system_prompt 由调用方按信号强度选档（纯闲聊 / 探需），
        话术本身集中在 agent/prompts.py，这里不写死。
        """
        system_msg = SystemMessage(content=system_prompt)
        history=state["messages"][-8:-1]
        return [system_msg] + history+[state["messages"][-1]]

    async def build_recommend_messages(self, state: AgentState, user_query: str, context: str)->List[BaseMessage]:
        """推荐用的消息组装。

        必须把检索到的真实商品拼进提示词——不拼就是让模型凭空编商品，
        推荐场景比问答更容易编得理所当然，用户也更难分辨。
        """
        current_summary = state.get("summary", "无")
        system_msg = SystemMessage(
            content=RECOMMEND_SYSTEM.format(context=context)
            + f"\n\n【历史会话摘要】\n{current_summary}"
        )
        history=state["messages"][-8:-1]
        return [system_msg] + history+[state["messages"][-1]]


    # async def build_compare_shop(self,state:AgentState,user_query:str,context:str)->List[BaseMessage]:
    #     system_msg=SystemMessage(content="你是一个善于比较的电商客服。请根据用户需要，比较两个商品的异同，并给出建议。")
    #
    #     history=state["messages"][-4:-1]
    #     return [system_msg] + history+[state["messages"][-1]]


    async def summarize_node(self,state:AgentState):
        messages=state.get("messages", [])
        old_summary=state.get("summary", "无")

        if len(messages) <= 8:
            return {}

        messages_to_summarize = messages[:-8]
        system_promopt= f"""
            你是一个严谨的电商数据摘要助手。
            这是之前的会话摘要：                    
            【{old_summary}】

            请阅读以下最新的对话记录，并将其与旧摘要合并，生成一段新的、不超过 100 字的摘要。
            【绝对红线规则】
            1. 必须包含用户关注的具体商品（带编号）、价格、库存或发货时间。
            2. 这些数字和关键事实绝对不允许改写、省略或模糊化。
            3. 保持客观，只记录事实。

            【最新对话记录】
            """
        for msg in messages_to_summarize:
            if not isinstance(msg,SystemMessage):
                role ="用户" if isinstance(msg,HumanMessage)else "客服"
                system_promopt+=f"{role}：{msg.content}\n"
        response=await self.summarizer_client.ainvoke([SystemMessage(content=system_promopt)])

        return{
            "summary": response.content.strip(),
        }

memory_manager = MemoryManager()



