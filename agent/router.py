import os
from typing import get_args
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from agent.deps import INTERNAL_TAG
from agent.prompts import ROUTER_SYSTEM, SIGNAL_JUDGE_SYSTEM
from utils.config_tool import get_app_config
from utils.log_tool import get_loger

logger = get_loger("agent.router")

#⚠️ 这里要和 main_agent 的 ROUTE_MAP 保持一致。
#多出来的意图不会崩（main_agent 有兜底），但会走兜底路径，等于这条意图没真正实现。
RouteName = Literal["闲聊", "业务咨询","概览筛选","商品对比","订单查询","商品推荐","越权拒答"]

#购买意向信号。这些名字必须是模型能 unambiguously 判断的业务语义，
#「信号1~信号10」这种占位符模型无从选择，判断准确率会塌掉。
SignalName = Literal[
    "购买意向", "品类偏好", "价格敏感", "场景需求",
    "品牌倾向", "对比犹豫", "售后顾虑", "明确拒绝",
]
SIGNAL_NAMES: tuple[str, ...] = get_args(SignalName)
#没命中任何信号的取值
SIGNAL_NONE = "无"
#有意愿的起点：达到这个值，闲聊回复要换成「探需」话术，主动把需求问清楚。
#这是成交前最关键的一步，憋到推荐阈值才开口就浪费了中间几轮。
SIGNAL_PROBE_THRESHOLD = 4
#强度达到这个值就触发推荐。注意 prompt 里的范围是 0-10，别写成 0-3
SIGNAL_THRESHOLD = 7


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


class SignalRouter:
    """购买意向信号判断器。

    只在闲聊链路里调用：每轮闲聊更新一次 {信号, 强度}，强度攒到 SIGNAL_THRESHOLD
    就触发推荐。它不参与意图分流——信号是「累积状态」，不是「这一轮该干什么」。
    """

    def __init__(self):
        self.client = init_chat_model(
            model=APP_CFG.get("llm.signal_model"),
            model_provider=APP_CFG.get("llm.model_provider"),
            temperature=APP_CFG.get("llm.signal_temperature"),
            api_key=os.getenv(APP_CFG.get("llm.api_key_env")),
            base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
        #⚠️ 必须打内部 tag：判断输出是「购买意向|6」这种结构化内容，
        #   不打 tag 会被 stream_mode="messages" 当成节点回答流给用户。
        ).with_config({"tags": [INTERNAL_TAG]})

    async def judge(self, state: MessagesState) -> dict:
        """判断本轮信号与强度，返回 {"signal": str, "level": int}"""
        user_query = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

        prev = state.get("signal_state")
        #判断失败时要能退回上轮状态，所以先把上一轮的值取出来
        prev_signal = prev.get("signal", SIGNAL_NONE) if prev else SIGNAL_NONE
        prev_level = prev.get("level", 0) if prev else 0

        router_prompt = SIGNAL_JUDGE_SYSTEM.format(
            prev_signal=prev_signal,
            prev_level=prev_level,
            user_query=user_query,
            #占位符 {signal_list} 必须传，漏了会直接 KeyError 崩掉整轮对话
            signal_list=" / ".join((*SIGNAL_NAMES, SIGNAL_NONE)),
        )

        raw = ""
        try:
            response = await self.client.ainvoke([
                SystemMessage(content=router_prompt),
                HumanMessage(content=user_query)
            ])
            raw = response.content.strip()

            name, level_str = raw.split("|")
            name, level = name.strip(), int(level_str.strip())
            #「无」也是合法输出（prompt 允许选它），漏掉会把所有"无关"的轮次
            #都判成异常、刷一堆 warning 日志，实际上它们完全正常
            if name not in (*SIGNAL_NAMES, SIGNAL_NONE) or not (0 <= level <= 10):
                raise ValueError(f"越界: {name}|{level}")
            #这两种情况强度强制为 0，不信模型返回的数字
            if name in (SIGNAL_NONE, "明确拒绝"):
                level = 0
        except Exception:
            #判断失败时保持上轮状态，不要归零——一次抖动不该清空用户攒了几轮的积累
            logger.warning(
                f"⚠️ 信号判断返回预期外内容 | 输入「{user_query[:30]}」| "
                f"原始输出: {raw[:80]!r} | 保持上轮 {prev_signal}/{prev_level}"
            )
            return {"signal": prev_signal, "level": prev_level}

        logger.info(f"📶 信号判断 | 上一轮 {prev_signal}/{prev_level} | 输入「{user_query[:30]}」-> {name}/{level}")
        return {"signal": name, "level": level}





router = IntentRouter()
signal_router = SignalRouter()


if __name__ == '__main__':
    import asyncio

    async def test():
        text_state={"messages": [HumanMessage(content="你好")]}
        result = await router.router(text_state)
        print(f"意图识别结果：{result}")

    asyncio.run(test())
