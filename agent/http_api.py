"""独立 HTTP 入口：匿名会话凭证、输入校验和 SSE 传输。"""

import asyncio
from contextlib import aclosing
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from time import monotonic
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import ClientDisconnect

from agent.main_agent import stream_reply
from agent.utils.log_tool import get_loger, set_session_id

app = FastAPI(title="电商客服 Agent", version="1.0.0")
logger = get_loger("agent.http")
ui_path = Path(__file__).with_name("http_ui")
app.mount("/ui", StaticFiles(directory=ui_path, check_dir=False), name="ui")
bearer = HTTPBearer(auto_error=False)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(strict=True, min_length=1, max_length=6000)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class SessionResponse(BaseModel):
    session_token: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def session_secret() -> str:
    secret = os.getenv("AGENT_SESSION_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(503, {"code": "CONFIGURATION_ERROR", "message": "会话密钥尚未配置。"})
    return secret


def session_signature(session_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), session_id.encode(), hashlib.sha256).hexdigest()


def authenticated_session(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    secret: Annotated[str, Depends(session_secret)],
) -> str:
    if credentials is None or re.fullmatch(r"[0-9a-f]{32}\.[0-9a-f]{64}", credentials.credentials) is None:
        raise HTTPException(401, {"code": "INVALID_SESSION", "message": "请新建会话后重试。"})
    session_id, signature = credentials.credentials.split(".")
    if not hmac.compare_digest(signature, session_signature(session_id, secret)):
        raise HTTPException(401, {"code": "INVALID_SESSION", "message": "请新建会话后重试。"})
    return session_id


@app.exception_handler(HTTPException)
async def http_error(request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc: RequestValidationError):
    return JSONResponse(
        {"error": {"code": "INVALID_REQUEST", "message": "请输入 1 至 6000 字符的非空文本。"}},
        status_code=422,
    )


def event(name: str, body: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


class ChatStreamResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        # Starlette 的 ASGI 2.4 分支只在发送时发现断开；等待模型期间也必须取消。
        async with aclosing(self.body_iterator):
            sending = asyncio.create_task(self.stream_response(send))
            disconnected = asyncio.create_task(self.listen_for_disconnect(receive))
            try:
                done, _ = await asyncio.wait((sending, disconnected), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except OSError as exc:
                raise ClientDisconnect() from exc
            finally:
                sending.cancel()
                disconnected.cancel()
                # 已观察完成任务的结果；收回取消中的任务后，外层 aclosing 才能安全关闭生成器。
                await asyncio.gather(sending, disconnected, return_exceptions=True)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(ui_path / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/api/sessions", status_code=201, response_model=SessionResponse,
    responses={503: {"model": ErrorResponse, "description": "CONFIGURATION_ERROR：服务尚未配置有效会话密钥。"}},
)
async def create_session(secret: Annotated[str, Depends(session_secret)]):
    session_id = uuid4().hex
    return {"session_token": f"{session_id}.{session_signature(session_id, secret)}"}


@app.post(
    "/api/chat",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE：meta → delta → done；服务异常以 error 事件结束。",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        401: {"model": ErrorResponse, "description": "INVALID_SESSION：会话凭证缺失、格式错误或签名无效。"},
        422: {"model": ErrorResponse, "description": "INVALID_REQUEST：请求须仅包含 1 至 6000 字符的非空 message。"},
        503: {"model": ErrorResponse, "description": "CONFIGURATION_ERROR：服务尚未配置有效会话密钥。"},
    },
)
async def chat(body: ChatRequest, session_id: Annotated[str, Depends(authenticated_session)]):
    request_id = uuid4().hex

    async def events():
        set_session_id(session_id)
        started = monotonic()
        outcome = "cancelled"
        logger.info("http_start request_id=%s", request_id)
        try:
            yield event("meta", {"request_id": request_id})
            async with aclosing(stream_reply(body.message, session_id, request_id=request_id)) as stream:
                async for text in stream:
                    yield event("delta", {"text": text})
            outcome = "completed"
            yield event("done", {"request_id": request_id})
        except Exception as exc:
            # HTTP 框架出口统一转换未知错误，取消不属于 Exception，必须原样传播。
            outcome = "error"
            logger.error("http_error request_id=%s error_type=%s", request_id, type(exc).__name__)
            yield event("error", {
                "code": "AGENT_FAILED", "message": "本轮服务异常，请稍后重试。", "request_id": request_id,
            })
        finally:
            logger.info("http_finish request_id=%s outcome=%s elapsed_ms=%.1f", request_id, outcome,
                        (monotonic() - started) * 1000)

    return ChatStreamResponse(events(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
