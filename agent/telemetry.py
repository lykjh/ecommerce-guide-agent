"""每轮独立的模型用量统计，只记录元数据，不记录提示词和输出。"""

import logging
import time
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class TurnUsage(BaseCallbackHandler):
    run_inline = True

    def __init__(self, request_id):
        self.request_id = request_id
        self.started = {}
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.unknown_usage = 0
        self.failures = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        stage = (kwargs.get("metadata") or {}).get("langgraph_node", "unknown")
        self.started[run_id] = (time.monotonic(), stage)
        self.calls += 1

    def on_llm_end(self, response, *, run_id, **kwargs):
        started, stage = self.started.pop(run_id, (time.monotonic(), "unknown"))
        usage = None
        finish_reason = None
        reasoning_tokens = None
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], "message", None)
            usage = getattr(message, "usage_metadata", None)
            finish_reason = (getattr(message, "response_metadata", None) or {}).get("finish_reason")
        if usage:
            incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
            reasoning_tokens = (usage.get("output_token_details") or {}).get("reasoning")
        else:
            usage = (response.llm_output or {}).get("token_usage") or {}
            incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        known = isinstance(incoming, int) and isinstance(outgoing, int)
        if known:
            self.input_tokens += incoming
            self.output_tokens += outgoing
        else:
            self.unknown_usage += 1
        logger.info("llm_usage request_id=%s stage=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s finish_reason=%s usage_known=%s duration_ms=%d",
                    self.request_id, stage, incoming, outgoing, reasoning_tokens, finish_reason,
                    known, (time.monotonic() - started) * 1000)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.started.pop(run_id, None)
        self.failures += 1
        self.unknown_usage += 1
        logger.warning("llm_failure request_id=%s error_type=%s", self.request_id, type(error).__name__)

    def finish(self):
        logger.info("turn_usage request_id=%s calls=%d input_tokens=%d output_tokens=%d unknown_usage=%d failures=%d unfinished=%d",
                    self.request_id, self.calls, self.input_tokens, self.output_tokens,
                    self.unknown_usage, self.failures, len(self.started))
