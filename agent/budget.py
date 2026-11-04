"""本轮共享模型预算；调用前原子预留，未知用量保守计费。"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import threading
import time
import inspect

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.config import ensure_config, merge_configs

logger = logging.getLogger(__name__)
CURRENT_BUDGET = ContextVar("agent_turn_budget", default=None)
_QUERY_SLOTS = threading.BoundedSemaphore(10)
BUDGET_REPLY = "本轮处理已停止，未完成的部分没有继续执行。请缩小问题范围后重试。"


class BudgetStop(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def estimate_input(messages, schema=None):
    # shortcut: 未绑定供应商 tokenizer，按UTF-8字节及协议余量保守预留；上线前校准。
    payload = [{"role": m.type, "content": m.content} for m in messages]
    data = {"messages": payload}
    if schema is not None:
        data["schema"] = schema.model_json_schema()
    return len(json.dumps(data, ensure_ascii=False).encode("utf-8")) + 128


class TurnBudget:
    MAX_CALLS = 8
    MAX_TOKENS = 24000
    MAX_INPUT = 8000
    MAX_OUTPUT = 1200
    TURN_SECONDS = 30
    CALL_SECONDS = 10
    ANSWER_TOKENS = 6000
    ANSWER_SECONDS = 5
    MAX_QUERIES = 6
    QUERY_SECONDS = 5

    def __init__(self, request_id):
        self.request_id = request_id
        self.deadline = time.monotonic() + self.TURN_SECONDS
        self.lock = threading.Lock()
        self.calls = 0
        self.spent = 0
        self.reserved = 0
        self.unknown = 0
        self.repairs = 0
        self.queries = 0
        self.answer_started = False
        self.stopped = None

    def stop(self, reason):
        with self.lock:
            self.stopped = self.stopped or reason
        logger.warning("budget_stop request_id=%s reason=%s", self.request_id, self.stopped)

    def reserve(self, incoming, outgoing, purpose):
        with self.lock:
            remaining = self.deadline - time.monotonic()
            keep_answer = not self.answer_started and purpose != "answer"
            reserve_tokens = self.ANSWER_TOKENS if keep_answer else 0
            reserve_calls = 1 if keep_answer else 0
            reason = self.stopped
            if not reason and remaining <= (self.ANSWER_SECONDS if keep_answer else 0):
                reason = "deadline"
            if not reason and (incoming > self.MAX_INPUT or outgoing > self.MAX_OUTPUT):
                reason = "single_call_tokens"
            if not reason and self.calls + 1 + reserve_calls > self.MAX_CALLS:
                reason = "model_calls"
            if not reason and self.spent + self.reserved + incoming + outgoing + reserve_tokens > self.MAX_TOKENS:
                reason = "tokens"
            if not reason and purpose == "repair" and self.repairs >= 1:
                reason = "repair_calls"
            if reason:
                self.stopped = reason
                logger.warning("budget_stop request_id=%s reason=%s", self.request_id, reason)
                raise BudgetStop(reason)
            self.calls += 1
            self.repairs += purpose == "repair"
            self.answer_started |= purpose == "answer"
            reservation = incoming + outgoing
            self.reserved += reservation
            logger.info("budget_reserve request_id=%s purpose=%s call=%d estimated_input=%d reserved_tokens=%d",
                        self.request_id, purpose, self.calls, incoming, reservation)
            return reservation, min(self.CALL_SECONDS, remaining - (self.ANSWER_SECONDS if keep_answer else 0))

    def settle(self, reservation, actual):
        with self.lock:
            self.reserved -= reservation
            self.spent += reservation if actual is None else actual
            self.unknown += actual is None
            if self.spent + self.reserved > self.MAX_TOKENS:
                self.stopped = "tokens"
        logger.info("budget_settle request_id=%s charged_tokens=%d estimated=%s spent=%d reserved=%d",
                    self.request_id, reservation if actual is None else actual, actual is None, self.spent, self.reserved)

    def finish(self):
        logger.info("budget_finish request_id=%s calls=%d queries=%d charged_tokens=%d unknown_usage=%d reserved=%d reason=%s",
                    self.request_id, self.calls, self.queries, self.spent, self.unknown, self.reserved, self.stopped or "completed")


@contextmanager
def budget_scope(budget):
    token = CURRENT_BUDGET.set(budget)
    try:
        yield
    finally:
        CURRENT_BUDGET.reset(token)


class _CallUsage(BaseCallbackHandler):
    run_inline = True

    def __init__(self):
        self.total = None

    def on_llm_end(self, response, **kwargs):
        message = getattr(response.generations[0][0], "message", None) if response.generations and response.generations[0] else None
        usage = getattr(message, "usage_metadata", None) or {}
        incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
        if incoming is None or outgoing is None:
            usage = (response.llm_output or {}).get("token_usage") or {}
            incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(incoming) is int and type(outgoing) is int and incoming >= 0 and outgoing >= 0:
            self.total = incoming + outgoing


async def invoke_model(model, messages, *, purpose="analysis", max_output=1200, schema=None, config=None, timeout=None):
    budget = CURRENT_BUDGET.get()
    # 独立诊断脚本不强制建立整轮；正式入口必须绑定共享预算。
    if budget is None:
        return await asyncio.wait_for(model.ainvoke(messages, config=config), timeout=timeout or TurnBudget.CALL_SECONDS)
    reservation, remaining = budget.reserve(estimate_input(messages, schema), max_output, purpose)
    timeout = min(timeout or TurnBudget.CALL_SECONDS, remaining)
    usage = _CallUsage()
    call_config = merge_configs(ensure_config(config), {"callbacks": [usage]})
    try:
        async with asyncio.timeout(timeout):
            return await model.ainvoke(messages, config=call_config)
    except TimeoutError:
        budget.stop("model_timeout")
        raise BudgetStop("model_timeout") from None
    except asyncio.CancelledError:
        budget.stop("deadline" if time.monotonic() >= budget.deadline else "cancelled")
        raise
    finally:
        budget.settle(reservation, usage.total)


async def invoke_query(function, *args, **kwargs):
    """同步查询进线程；异步查询留在原事件循环，取消时释放查询名额。"""
    budget = CURRENT_BUDGET.get()
    timeout = TurnBudget.QUERY_SECONDS
    if budget is not None:
        with budget.lock:
            remaining = budget.deadline - time.monotonic() - (0 if budget.answer_started else budget.ANSWER_SECONDS)
            reason = budget.stopped or ("query_calls" if budget.queries >= budget.MAX_QUERIES else None)
            if remaining <= 0:
                reason = "deadline"
            if reason:
                budget.stopped = reason
                logger.warning("budget_stop request_id=%s reason=%s", budget.request_id, reason)
                raise BudgetStop(reason)
            budget.queries += 1
            timeout = min(budget.QUERY_SECONDS, remaining)
    if not _QUERY_SLOTS.acquire(blocking=False):
        if budget:
            budget.stop("query_busy")
        raise BudgetStop("query_busy")

    def work():
        try:
            return function(*args, **kwargs)
        finally:
            _QUERY_SLOTS.release()

    is_async = inspect.iscoroutinefunction(function)
    if not is_async:
        # 同步线程超时后继续占用名额，直到网络请求真实结束。
        task = asyncio.create_task(asyncio.to_thread(work))
        task.add_done_callback(lambda finished: None if finished.cancelled() else finished.exception())
    started = time.monotonic()
    status = "completed"
    try:
        if is_async:
            return await asyncio.wait_for(function(*args, **kwargs), timeout=timeout)
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except TimeoutError:
        status = "timeout" if is_async else "timeout_background_pending"
        if budget:
            budget.stop("query_timeout")
        raise BudgetStop("query_timeout") from None
    except asyncio.CancelledError:
        status = "cancelled" if is_async else "cancelled_background_pending"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        if is_async:
            _QUERY_SLOTS.release()
        logger.info("query_finish request_id=%s status=%s duration_ms=%d",
                    budget.request_id if budget else "standalone", status, (time.monotonic() - started) * 1000)
