from openai import APIError
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
import asyncio
import json
import logging
import re
from typing import get_args
from typing import Literal, Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from langchain_core.messages import HumanMessage, AIMessage
from agent.deps import build_client, INTERNAL_TAG
from agent.budget import BudgetStop, invoke_model
from agent.memory import memory_manager
from agent.state import AgentState
from agent.prompts import ROUTER_SYSTEM, SIGNAL_JUDGE_SYSTEM
from agent.utils.config_tool import get_app_config

logger = logging.getLogger(__name__)

RouteName = Literal[
    "闲聊", "业务咨询", "概览筛选", "商品对比", "订单查询", "商品推荐",
    "政策问答", "转人工", "澄清", "越权拒答", "超出范围",
]
ROUTE_NAMES = get_args(RouteName)


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    intent: RouteName
    subjects: list[Annotated[str, StringConstraints(min_length=1, max_length=120, pattern=r"^[^\r\n]*\S[^\r\n]*$")]] = Field(max_length=8)
    needs_clarification: bool


ROUTE_MAP = {
    "闲聊": "chat_node", "业务咨询": "business_node", "概览筛选": "total_node",
    "商品对比": "compare_shop_node", "商品推荐": "recommend_node",
    **{name: "route_reply_node" for name in ("订单查询", "政策问答", "转人工", "澄清", "越权拒答", "超出范围", "识别失败")},
}


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


def normalize_list_subjects(subjects, messages):
    """只去掉误复制的列表引导词；校正目标必须来自最近明确列举的原文。"""
    current = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or not isinstance(message.content, str):
            continue
        match = re.fullmatch(r"(?:候选(?:依次为|为)?|找到|推荐如下)[：: ]*(.+)", message.content.strip())
        if not match:
            continue
        candidates = [name.strip() for name in re.split(r"[、，,]", match.group(1).rstrip("。"))]
        if not 2 <= len(candidates) <= 8 or any(not name or len(name) > 120 for name in candidates):
            continue
        normalized = []
        for subject in subjects:
            stripped = re.sub(r"^(?:候选(?:依次为|为)?|找到|推荐如下)[：: ]*", "", subject)
            normalized.append(stripped if subject not in current and subject not in candidates and stripped in candidates else subject)
        return normalized
    return subjects


class IntentRouter:
    TIMEOUT = 10

    def __init__(self, client=None):
        self.client = client

    async def decision(self, state: AgentState) -> tuple[str, list[str]]:
        messages = state.get("messages", [])
        if (not messages or not isinstance(messages[-1], HumanMessage)
                or not isinstance(messages[-1].content, str)
                or not messages[-1].content.strip()):
            return "澄清", []
        if self.client is None:
            self.client = build_client(
                "llm.router_model", 0, max_tokens=256,
                model_kwargs={"response_format": {"type": "json_schema", "json_schema": {
                    "name": "route_decision", "strict": True, "schema": RouteDecision.model_json_schema()}}},
            )
        has_history = bool(state.get("summary")
                           or state.get("task_context")
                           or state.get("candidate_products")
                           or len(memory_manager.recent_messages(messages)) > 1)
        prompt = ROUTER_SYSTEM.format(
            intent_list=" / ".join(ROUTE_NAMES),
            history_status="存在，需核实是否包含必要对象" if has_history else "没有，当前消息是唯一业务信息",
        )
        context_messages = memory_manager.build_messages(state, prompt, purpose="route")
        try:
            response = await invoke_model(self.client,
                context_messages, max_output=256, schema=RouteDecision, config={"tags": [INTERNAL_TAG]},
                timeout=self.TIMEOUT)
        except (APIError, TimeoutError) as exc:
            logger.warning("路由模型调用失败：%s", type(exc).__name__)
            return "识别失败", []
        if response.response_metadata.get("finish_reason") == "length":
            logger.warning("route_failure reason=output_truncated")
            return "识别失败", []
        if not isinstance(response.content, str):
            return "识别失败", []
        try:
            decision = RouteDecision.model_validate_json(response.content)
        except ValidationError as exc:
            logger.warning("路由输出解析失败：%s", type(exc).__name__)
            return "识别失败", []
        intent, subjects = decision.intent, decision.subjects
        normalized = normalize_list_subjects(subjects, context_messages[1:])
        if normalized != subjects:
            logger.info("route_subjects_normalized reason=list_heading count=%d", len(subjects))
        subjects = normalized
        # 拒答和人工优先级不能被对象缺失降为普通澄清。
        if intent not in ("越权拒答", "转人工"):
            if intent in ("业务咨询", "商品对比"):
                if decision.needs_clarification:
                    return "澄清", []
                evidence = [m.content for m in context_messages[1:]]
                if any(not any(subject in text for text in evidence) for subject in subjects):
                    return "澄清", []
                # 先验证原对象来源，再分离显式编号与后附属性；含多个数字的对象不擅自取第一个。
                for index, subject in enumerate(subjects):
                    match = re.fullmatch(r"商品编号[：:]?\s*([1-9][0-9]*)\D*", subject)
                    if match:
                        subjects[index] = match.group(1)
                names = set(subjects)
                ambiguous = len(names) < 2 if intent == "商品对比" else len(names) != 1
                attribute_only = any(re.fullmatch(r"(?:黑|白|红|蓝|绿|黄|紫|粉|灰|银|金)色(?:版|款)?", name)
                                     or name in {"它", "这个", "那个", "这款", "那款", "第二个"} for name in names)
                if ambiguous or attribute_only:
                    return "澄清", []
        logger.info("意图识别完成：%s", intent)
        return intent, list(dict.fromkeys(subjects)) if intent in ("业务咨询", "商品对比") else []

    async def router(self, state: AgentState) -> str:
        intent, _ = await self.decision(state)
        return intent

    async def node(self, state: AgentState) -> dict:
        category = state.get("category_context", {})
        query = state["messages"][-1].content.strip()
        action = {}
        task = state.get("task_context", {})
        if query in ("浏览分类", "查看分类", "按分类浏览", "查看商品分类"):
            return {"intent": "概览筛选", "final_reply": "", "route_subjects": [],
                    "category_action": {"kind": "browse_root"}}
        if (task.get("intent") in ("商品推荐", "概览筛选") and state.get("product_page")
                and re.fullmatch(r"(?:下一批|下一页|更多候选)[。！!]?[?？]?", query)):
            return {"intent": task["intent"], "final_reply": "", "route_subjects": [],
                    "category_action": {"kind": "next_page"}}
        if category.get("pending"):
            choice = re.fullmatch(r"(?:选(?:择)?\s*)?第?([1-9][0-9]*|[一二三四五六七八九十])(?:个|项)?[。！!]?", query)
            browse = re.fullmatch(r"(?:查看|展开)第?([1-9][0-9]*|[一二三四五六七八九十])(?:个|项)?(?:的)?子分类[。！!]?", query)
            if choice or browse:
                value = (browse or choice).group(1)
                if value.isdecimal():
                    count = len(category["candidates"])
                    index = int(value) if len(value) <= len(str(count)) else count + 1
                else:
                    index = "一二三四五六七八九十".index(value) + 1
                action = {"kind": "browse" if browse else "select", "index": index - 1}
            elif query == "按关键词搜索":
                action = {"kind": "keyword"}
            else:
                exact = [i for i, candidate in enumerate(category["candidates"]) if query == candidate["name"]]
                if len(exact) == 1:
                    action = {"kind": "select", "index": exact[0]}
        if action:
            return {"intent": state["task_context"]["intent"], "final_reply": "", "route_subjects": [],
                    "category_action": action}
        if query in state.get("task_context", {}).get("color_choices", []):
            return {"intent": state["task_context"]["intent"], "final_reply": "", "route_subjects": [],
                    "category_action": {}}
        intent, subjects = await self.decision(state)
        update = {"intent": intent, "final_reply": "", "route_subjects": subjects, "category_action": {}}
        product_intents = ("商品推荐", "概览筛选")
        if (state.get("task_context") and intent in product_intents
                and state["task_context"].get("intent") in product_intents
                and intent != state["task_context"]["intent"]):
            update["task_context"] = {**state["task_context"], "intent": intent}
        elif state.get("task_context") and intent not in (state["task_context"].get("intent"), "澄清", "识别失败"):
            update["task_context"] = {}
        if category.get("pending") and intent not in product_intents:
            update["category_context"] = {}
            update["product_page"] = {}
        return update


def route_with_fallback(state: AgentState) -> str:
    intent = state.get("intent", "识别失败")
    return intent if intent in ROUTE_MAP else "识别失败"


class SignalRouter:
    """购买意向信号判断器。

    只在闲聊链路里调用：每轮闲聊更新一次 {信号, 强度}，强度攒到 SIGNAL_THRESHOLD
    就触发推荐。它不参与意图分流——信号是「累积状态」，不是「这一轮该干什么」。
    """

    def __init__(self):
        self.client = None

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
            user_query="见最后一条用户消息",
            #占位符 {signal_list} 必须传，漏了会直接 KeyError 崩掉整轮对话
            signal_list=" / ".join((*SIGNAL_NAMES, SIGNAL_NONE)),
        )

        raw = ""
        try:
            if self.client is None:
                self.client = build_client(
                    "llm.signal_model", APP_CFG.get("llm.signal_temperature"),
                ).with_config({"tags": [INTERNAL_TAG]})
            response = await invoke_model(self.client, memory_manager.build_messages(state, router_prompt, purpose="signal"))
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
        except BudgetStop:
            raise
        except (APIError, TimeoutError, ValueError) as exc:
            #判断失败时保持上轮状态，不要归零——一次抖动不该清空用户攒了几轮的积累
            logger.warning("信号判断失败，保持上轮状态")
            return {"signal": prev_signal, "level": prev_level}

        logger.info("信号判断完成 level=%d", level)
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
