"""独立 HTTP 入口：匿名会话凭证、输入校验和 SSE 传输。"""

import asyncio
from contextlib import aclosing, asynccontextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from time import monotonic
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import ClientDisconnect

from agent.main_agent import stream_reply
from agent.utils.log_tool import get_loger, set_session_id

from agent.deps import DB_PATH
from agent.identity import MALL_AUTH_COOKIE, IdentityClient, IdentityError, validate_url
from agent.nodes.orders import OrderTurns
from agent.session_store import SessionStore, SessionStoreError
from agent.tool.orders import OrderAccess, OrderAuthExpired


@asynccontextmanager
async def lifespan(app):
    app.state.owners = SessionStore(Path(DB_PATH).with_suffix('.sessions.db'))
    app.state.identity = IdentityClient()
    try:
        await app.state.owners.open()
    except SessionStoreError:
        logger.error("session_access result=unavailable code=SESSION_STORE_UNAVAILABLE")
    try:
        yield
    finally:
        await app.state.owners.close()


app = FastAPI(title="电商客服 Agent", version="1.1.0", lifespan=lifespan)
order_turns = OrderTurns()
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
        raise HTTPException(503, {"code": "CONFIGURATION_ERROR", "message": "客服服务配置未完成。"})
    return secret


def session_signature(session_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), session_id.encode(), hashlib.sha256).hexdigest()


def authenticated_session(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    request: Request,
) -> str:
    secret = request.state.secret
    if credentials is None or re.fullmatch(r"[0-9a-f]{32}\.[0-9a-f]{64}", credentials.credentials) is None:
        raise HTTPException(401, {"code": "INVALID_SESSION", "message": "请新建会话后重试。"})
    session_id, signature = credentials.credentials.split(".")
    if not hmac.compare_digest(signature, session_signature(session_id, secret)):
        raise HTTPException(401, {"code": "INVALID_SESSION", "message": "请新建会话后重试。"})
    return session_id


ERRORS = {
    'CONFIGURATION_ERROR': (503, '客服服务配置未完成。'),
    'ORIGIN_NOT_ALLOWED': (403, '请求来源不受支持。'),
    'IDENTITY_UNAVAILABLE': (503, '暂时无法确认登录状态，请稍后重试。'),
    'IDENTITY_BAD_RESPONSE': (502, '登录状态响应异常，请稍后重试。'),
    'IDENTITY_BUSY': (503, '登录状态校验繁忙，请稍后重试。'),
    'SESSION_STORE_UNAVAILABLE': (503, '会话服务暂时不可用，请稍后重试。'),
    'INVALID_SESSION': (401, '请新建会话后重试。'),
    'SESSION_IDENTITY_CHANGED': (409, '登录身份已变化，请在新会话中重新发送。'),
}


def reject(code):
    status, message = ERRORS[code]
    raise HTTPException(status, {'code': code, 'message': message})


class SourceBoundary:
    """Validate at ingress without adding a cancellation task group around SSE."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        request.state.request_id = uuid4().hex
        if request.method == 'POST' and request.url.path.removeprefix(scope.get('root_path', '')) in ('/api/sessions', '/api/chat'):
            try:
                origin = validate_url(os.getenv('AGENT_PUBLIC_ORIGIN', ''), origin=True)
                if request.headers.get('origin') != origin:
                    reject('ORIGIN_NOT_ALLOWED')
                request.state.mall_base = validate_url(os.getenv('MALL_BASE_URL', ''))
                request.state.secret = session_secret()
            except IdentityError as exc:
                status, message = ERRORS[exc.code]
                await JSONResponse({'error': {'code': exc.code, 'message': message}}, status_code=status)(scope, receive, send)
                return
            except HTTPException as exc:
                await JSONResponse({'error': exc.detail}, status_code=exc.status_code)(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(SourceBoundary)


async def current_identity(request, operation):
    started = monotonic()
    result, code = 'cancelled', '-'
    try:
        owner = await request.app.state.identity.resolve(request.cookies.get(MALL_AUTH_COOKIE),
                                                        request.state.mall_base, request.state.secret)
        result = owner.kind
        return owner
    except IdentityError as exc:
        result, code = 'failed', exc.code
        reject(code)
    finally:
        logger.info('identity_check request_id=%s operation=%s result=%s code=%s elapsed_ms=%.1f',
                    request.state.request_id, operation, result, code, (monotonic()-started)*1000)


async def owned_session(request: Request, session_id: str):
    started = monotonic()
    code = '-'
    try:
        owner = await request.app.state.owners.get(session_id)
        if owner is None:
            reject('INVALID_SESSION')
        current = await current_identity(request, 'chat')
        if current != owner:
            order_turns.sessions.pop(session_id, None)
            reject('SESSION_IDENTITY_CHANGED')
        return current
    except SessionStoreError:
        code = 'SESSION_STORE_UNAVAILABLE'
        reject(code)
    except HTTPException as exc:
        code = exc.detail['code']
        raise
    finally:
        logger.info('session_access request_id=%s operation=chat code=%s elapsed_ms=%.1f',
                    request.state.request_id, code, (monotonic()-started)*1000)


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
    responses={status: {"model": ErrorResponse, "description": " / ".join(code for code, value in ERRORS.items() if value[0] == status)} for status in (403, 502, 503)},
)
async def create_session(request: Request):
    secret = request.state.secret
    owner = await current_identity(request, 'create')
    session_id = uuid4().hex
    started = monotonic()
    code = '-'
    try:
        await request.app.state.owners.create(session_id, owner)
    except SessionStoreError:
        code = 'SESSION_STORE_UNAVAILABLE'
        reject(code)
    finally:
        logger.info('session_access request_id=%s operation=create code=%s elapsed_ms=%.1f',
                    request.state.request_id, code, (monotonic()-started)*1000)
    return {"session_token": f"{session_id}.{session_signature(session_id, secret)}"}


@app.post(
    "/api/chat",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE：meta → delta → done；服务异常以 error 事件结束。",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        **{status: {"model": ErrorResponse, "description": " / ".join(code for code, value in ERRORS.items() if value[0] == status)} for status in (403, 409, 502)},
        401: {"model": ErrorResponse, "description": "INVALID_SESSION：会话凭证缺失、格式错误或签名无效。"},
        422: {"model": ErrorResponse, "description": "INVALID_REQUEST：请求须仅包含 1 至 6000 字符的非空 message。"},
        503: {"model": ErrorResponse, "description": "CONFIGURATION_ERROR / IDENTITY_UNAVAILABLE / IDENTITY_BUSY / SESSION_STORE_UNAVAILABLE"},
    },
)
async def chat(request: Request, body: ChatRequest, session_id: Annotated[str, Depends(authenticated_session)]):
    owner = await owned_session(request, session_id)
    request_id = request.state.request_id
    access = OrderAccess(owner, request.state.mall_base, request.cookies.get(MALL_AUTH_COOKIE),
                         request.state.secret, request.app.state.identity)

    async def events():
        set_session_id(session_id)
        started = monotonic()
        outcome = "cancelled"
        logger.info("http_start request_id=%s", request_id)
        try:
            yield event("meta", {"request_id": request_id})
            options = {"product_links": []} if request.query_params.get("product_links") == "1" else {}
            async with aclosing(stream_reply(body.message, session_id, request_id=request_id,
                               order_handler=lambda: order_turns.handle(body.message, session_id, access),
                               **options)) as stream:
                async for text in stream:
                    yield event("delta", {"text": text})
            outcome = "completed"
            done = {"request_id": request_id}
            if options.get("product_links"):
                done["product_links"] = options["product_links"]
            yield event("done", done)
        except OrderAuthExpired:
            outcome = "error"
            yield event("error", {"code": "ORDER_AUTH_EXPIRED", "message": "商城登录已失效，请重新登录后查询。",
                                  "request_id": request_id})
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
                              headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})
