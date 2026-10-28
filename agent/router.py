import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from agent.prompts import ROUTER_SYSTEM
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

        #提示词在 agent/prompts.py，这里只负责调模型和清洗结果
        router_prompt=ROUTER_SYSTEM.format(user_query=user_query)

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
