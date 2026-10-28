import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from utils.config_tool import get_app_config
from utils.log_tool import get_loger

logger = get_loger("agent.router")

#⚠️ 这里要和 main_agent 的 ROUTE_MAP 保持一致。
#多出来的意图不会崩（main_agent 有兜底），但会走兜底路径，等于这条意图没真正实现。
RouteName = Literal["闲聊", "业务咨询","概览筛选","商品对比","订单查询","商品推荐","越权拒答"]

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
        意图名称必须从["闲聊", "业务咨询","概览筛选","商品对比","订单查询","商品推荐","越权拒答"]中选择。
        例如
        判断用户最新输入属于哪种意图：
            - 闲聊/问候（如“你好”、“你是谁”）
            - 业务咨询（询问具体或大概商品价格、库存、发货、售后等）
            - 概览筛选（如“我想看手机”、“我想看笔记本电脑”）
            - 商品对比（如“比一下iPhone和华为P30”）
            - 订单查询（如“查询订单号123456789”的订单状态）
            - 商品推荐（如“推荐我一部手机”）
            - 越权拒答（如“查询订单号123456789”的订单状态）
        用户输入：{user_query}
        意图名称：
        请严格只回复一个意图名称，不要添加其他文字。
        """

        response=await self.client.ainvoke([SystemMessage(content=router_prompt), HumanMessage(content=user_query)])
        #模型偶尔会带上标点、引号或多余解释，这里只做最小清洗，不做模糊匹配——
        #匹配不上就如实记日志走兜底，别猜。
        intent = response.content.strip().strip("。.！!，,＂\"'")

        known = {"闲聊", "业务咨询", "概览筛选", "商品对比", "订单查询", "商品推荐", "越权拒答"}

        if intent in known:
            logger.info(f"🧭 意图识别 | 输入「{user_query[:30]}」-> {intent}")
            return intent

        #模型吐了预期之外的内容（可能是整句话而非意图名），必须记下来，否则线上无从排查
        logger.warning(
            f"⚠️ 意图识别返回了预期外的内容，将走「越权拒答」| "
            f"输入「{user_query[:30]}」| 模型原始输出: {response.content[:80]!r}"
        )
        return "越权拒答"


router = IntentRouter()

if __name__ == '__main__':
    import asyncio

    async def test():
        text_state={"messages": [HumanMessage(content="你好")]}
        result = await router.router(text_state)
        print(f"意图识别结果：{result}")

    asyncio.run(test())
