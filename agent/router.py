import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from utils.config_tool import get_app_config

RouteName = Literal["闲聊", "业务咨询", "其他"]

APP_CFG = get_app_config()
class IntentRouter:
    def __init__(self):
        self.client=init_chat_model(
            model=APP_CFG.get("llm.router_model"),
            model_provider=APP_CFG.get("llm.model_provider"),
            temperature=APP_CFG.get("llm.router_temperature"),
            api_key=os.getenv(APP_CFG.get("llm.api_key_env")),
            base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
        )

    async def router(self,state:MessagesState)->RouteName:
        """对外暴露的唯一方法，直接交给 LangGraph 路由节点使用"""
        user_query=next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),"")

        router_prompt=f"""你是一个意图识别器。
        你的任务是根据用户输入，判断用户意图，并返回意图名称。
        意图名称必须从["闲聊", "业务咨询", "其他"]中选择。
        例如
        判断用户最新输入属于哪种意图：
            - 闲聊/问候（如“你好”、“你是谁”）
            - 业务咨询（询问商品价格、库存、发货、售后等）
            - 其他（无法判断或不属于上述意图）
        用户输入：{user_query}
        意图名称：
        请严格只回复一个意图名称，不要添加其他文字。
        """

        response=await self.client.ainvoke([SystemMessage(content=router_prompt), HumanMessage(content=user_query)])
        intent = response.content.strip()

        if intent == "闲聊":
            return "闲聊"
        elif intent == "业务咨询":
            return "业务咨询"
        else:
            return "其他"

router = IntentRouter()

if __name__ == '__main__':
    import asyncio

    async def test():
        text_state={"messages": [HumanMessage(content="你好")]}
        result = await router.router(text_state)
        print(f"意图识别结果：{result}")

    asyncio.run(test())
