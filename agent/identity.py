"""Request-scoped mall identity; credentials never enter graph state."""
import asyncio
from dataclasses import dataclass
import hashlib
import hmac
from http.client import HTTPException
import json
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, Field, ValidationError

MALL_AUTH_COOKIE = 'MALLAUTH'


class IdentityError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Identity:
    kind: str
    user_id: str | None = None
    login_fingerprint: str | None = None


class Envelope(BaseModel):
    code: int = Field(strict=True)
    message: str
    data: object


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_url(value: str, *, origin=False) -> str:
    try:
        parsed = urlsplit(value)
        parsed.port
        valid = (parsed.scheme in ('http', 'https') and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and not any(c.isspace() for c in value)
                 and (not origin or not parsed.path))
    except ValueError:
        valid = False
    if not valid:
        raise IdentityError('CONFIGURATION_ERROR')
    return value


class IdentityClient:
    def __init__(self):
        self.slots = threading.BoundedSemaphore(4)

    def _query(self, cookie, base, secret):
        try:
            try:
                response = build_opener(NoRedirect()).open(
                    Request(base.rstrip('/') + '/users/me', headers={'Cookie': MALL_AUTH_COOKIE + '=' + cookie}), timeout=5)
            except HTTPError as exc:
                response = exc
            with response:
                status = response.status
                if status == 403 or status == 429 or status >= 500 or 300 <= status < 400:
                    raise IdentityError('IDENTITY_UNAVAILABLE')
                if status not in (200, 401):
                    raise IdentityError('IDENTITY_BAD_RESPONSE')
                envelope = Envelope.model_validate(json.loads(response.read()))
            if envelope.code != status:
                raise IdentityError('IDENTITY_BAD_RESPONSE')
            if status == 401:
                return Identity('guest')
            data = envelope.data
            if not isinstance(data, dict) or type(data.get('id')) is not int or data['id'] <= 0:
                raise IdentityError('IDENTITY_BAD_RESPONSE')
            fingerprint = hmac.new(secret.encode(), ('mall-login\0' + cookie).encode(), hashlib.sha256).hexdigest()
            return Identity('user', str(data['id']), fingerprint)
        except (URLError, OSError, HTTPException):
            raise IdentityError('IDENTITY_UNAVAILABLE') from None
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise IdentityError('IDENTITY_BAD_RESPONSE') from None

    async def resolve(self, cookie, base, secret):
        if cookie is None:
            return Identity('guest')
        if not self.slots.acquire(blocking=False):
            raise IdentityError('IDENTITY_BUSY')

        def work():
            try:
                return self._query(cookie, base, secret)
            finally:
                self.slots.release()

        task = asyncio.create_task(asyncio.to_thread(work))
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            return await asyncio.wait_for(asyncio.shield(task), 5)
        except TimeoutError:
            raise IdentityError('IDENTITY_UNAVAILABLE') from None
