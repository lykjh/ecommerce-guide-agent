"""独立 HTTP 入口契约；所有对话使用替身，不调用在线模型。"""
import asyncio
import json
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent import http_api, main_agent


SECRET = "http-test-session-secret-32-characters-minimum"


def events(response_text):
    """按公开 SSE 协议解析完整响应，多个场景共用。"""
    frames = []
    for frame in response_text.strip().split("\n\n"):
        fields = dict(line.split(":", 1) for line in frame.splitlines())
        frames.append((fields["event"].strip(), json.loads(fields["data"])))
    return frames


async def answer(*args, **kwargs):
    yield "耳机\n"
    yield "价格 299 元🙂"


class HttpApiTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"AGENT_SESSION_SECRET": SECRET, "AGENT_PUBLIC_ORIGIN": "http://testserver", "MALL_BASE_URL": "http://mall.invalid"})
        env.start()
        self.addCleanup(env.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db = patch.object(http_api, "DB_PATH", str(Path(directory.name)/"chat.db"))
        db.start()
        self.addCleanup(db.stop)
        self.client = TestClient(http_api.app, headers={"Origin":"http://testserver"})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.post("/api/sessions").json()["session_token"]
        self.headers = {"Authorization": "Bearer " + self.token}

    def test_health_and_ui_do_not_call_model(self):
        with patch("agent.http_api.stream_reply") as stream:
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})
            page = self.client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.headers["content-type"])
            for path in ("/ui/chat.js", "/ui/style.css"):
                with self.subTest(path=path):
                    asset = self.client.get(path)
                    self.assertEqual(asset.status_code, 200)
                    self.assertTrue(asset.content)
            stream.assert_not_called()

    def test_new_session_tokens_are_distinct_and_not_histories(self):
        with patch("agent.http_api.stream_reply") as stream:
            response = self.client.post("/api/sessions")
            self.assertEqual(response.status_code, 201)
            self.assertEqual(set(response.json()), {"session_token"})
            token = response.json()["session_token"]
            self.assertRegex(token, r"^[0-9a-f]{32}\.[0-9a-f]{64}$")
            self.assertNotEqual(token, self.token)
            stream.assert_not_called()

    def test_stream_frames_preserve_text_and_request_id(self):
        with patch("agent.http_api.stream_reply", side_effect=answer) as stream:
            response = self.client.post("/api/chat", headers=self.headers, json={"message": "推荐耳机"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(response.headers["cache-control"], "no-cache")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        frames = events(response.text)
        self.assertEqual([name for name, _ in frames], ["meta", "delta", "delta", "done"])
        self.assertEqual("".join(data["text"] for name, data in frames if name == "delta"), "耳机\n价格 299 元🙂")
        request_id = frames[0][1]["request_id"]
        self.assertTrue(request_id)
        self.assertEqual(frames[-1][1], {"request_id": request_id})
        stream.assert_called_once()
        args, kwargs = stream.call_args
        self.assertEqual(args[:2], ("推荐耳机", self.token.split(".")[0]))
        self.assertEqual(kwargs.get("request_id", args[2] if len(args) > 2 else None), request_id)

    def test_session_isolation_and_resume(self):
        another = self.client.post("/api/sessions").json()["session_token"]
        with patch("agent.http_api.stream_reply", side_effect=answer) as stream:
            for token in (self.token, another, self.token):
                response = self.client.post("/api/chat", headers={"Authorization": "Bearer " + token}, json={"message": "继续"})
                self.assertEqual(response.status_code, 200)
        ids = [call.args[1] for call in stream.call_args_list]
        self.assertEqual(ids[0], ids[2])
        self.assertNotEqual(ids[0], ids[1])

    def test_invalid_input_is_422_without_forwarding_or_echo(self):
        bodies = ({}, {"message": ""}, {"message": " \t\n"}, {"message": None},
                  {"message": 123}, {"message": ["hello"]}, {"message": "x" * 6001},
                  {"message": "private-input-marker", "thread_id": "victim"})
        with patch("agent.http_api.stream_reply") as stream:
            for body in bodies:
                with self.subTest(body_type=str(type(body.get("message")))):
                    response = self.client.post("/api/chat", headers=self.headers, json=body)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
                    self.assertNotIn("private-input-marker", response.text)
            malformed = self.client.post("/api/chat", headers={**self.headers, "Content-Type": "application/json"}, content='{"message": private-input-marker')
            self.assertEqual(malformed.status_code, 422)
            self.assertNotIn("private-input-marker", malformed.text)
            stream.assert_not_called()

    def test_exact_input_limit_is_accepted_without_trimming(self):
        message = " " + "字" * 5999
        with patch("agent.http_api.stream_reply", side_effect=answer) as stream:
            response = self.client.post("/api/chat", headers=self.headers, json={"message": message})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(stream.call_args.args[0], message)

    def test_invalid_or_forged_session_never_reaches_model(self):
        signature = self.token[-1]
        forged = self.token[:-1] + ("0" if signature != "0" else "1")
        for value in (None, "", "Basic " + self.token, "Bearer malformed", "Bearer " + forged, "Bearer " + self.token + "suffix"):
            with self.subTest(header=value), patch("agent.http_api.stream_reply") as stream:
                response = self.client.post("/api/chat", headers={} if value is None else {"Authorization": value}, json={"message": "hello"})
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["error"]["code"], "INVALID_SESSION")
                self.assertNotIn(self.token, response.text)
                stream.assert_not_called()

    def test_rotated_secret_invalidates_old_session(self):
        with patch.dict(os.environ, {"AGENT_SESSION_SECRET": "new-" + SECRET}), patch("agent.http_api.stream_reply") as stream:
            response = self.client.post("/api/chat", headers=self.headers, json={"message": "hello"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "INVALID_SESSION")
        stream.assert_not_called()

    def test_missing_or_short_secret_is_configuration_error(self):
        for secret in ("", "short"):
            with self.subTest(secret=secret), patch.dict(os.environ, {"AGENT_SESSION_SECRET": secret}), patch("agent.http_api.stream_reply") as stream:
                response = self.client.post("/api/sessions")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], "CONFIGURATION_ERROR")
                response = self.client.post("/api/chat", headers=self.headers, json={"message": "hello"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], "CONFIGURATION_ERROR")
                stream.assert_not_called()

    def test_stream_failure_is_terminal_and_redacted_in_response_and_log(self):
        async def failing(*args, **kwargs):
            yield "已生成部分"
            raise RuntimeError("secret-provider-response-marker")

        with patch("agent.http_api.stream_reply", side_effect=failing), self.assertLogs(http_api.logger.name, level="INFO") as logs:
            response = self.client.post("/api/chat", headers=self.headers, json={"message": "private-query-marker"})
        self.assertEqual(response.status_code, 200)
        frames = events(response.text)
        self.assertEqual([name for name, _ in frames], ["meta", "delta", "error"])
        self.assertEqual(frames[-1][1], {"code": "AGENT_FAILED", "message": "本轮服务异常，请稍后重试。", "request_id": frames[0][1]["request_id"]})
        combined = response.text + "\n".join(logs.output)
        for sensitive in (self.token, SECRET, "private-query-marker", "secret-provider-response-marker"):
            self.assertNotIn(sensitive, combined)
        self.assertIn(frames[0][1]["request_id"], "\n".join(logs.output))


class HttpDisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_while_model_waits_closes_stream(self):
        entered, closed = asyncio.Event(), asyncio.Event()

        async def waiting(*args, **kwargs):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "not reached"
            finally:
                closed.set()

        with patch.dict(os.environ, {"AGENT_SESSION_SECRET": SECRET, "AGENT_PUBLIC_ORIGIN": "http://testserver", "MALL_BASE_URL": "http://mall.invalid"}):
            from agent.identity import Identity
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            store = http_api.SessionStore(Path(directory.name)/"owners.db")
            await store.open()
            self.addAsyncCleanup(store.close)
            http_api.app.state.owners = store
            http_api.app.state.identity = http_api.IdentityClient()
            sid = "a" * 32
            await store.create(sid, Identity("guest"))
            token = sid + "." + http_api.session_signature(sid, SECRET)
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                     "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/chat",
                     "raw_path": b"/api/chat", "query_string": b"", "root_path": "",
                     "headers": [(b"origin", b"http://testserver"), (b"content-type", b"application/json"), (b"authorization", ("Bearer " + token).encode())],
                     "server": ("testserver", 80), "client": ("127.0.0.1", 12345)}
            sent = []
            first = True

            async def receive():
                nonlocal first
                if first:
                    first = False
                    return {"type": "http.request", "body": b'{"message":"hello"}', "more_body": False}
                await entered.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            with patch("agent.main_agent._get_ai_stream", side_effect=waiting):
                await asyncio.wait_for(http_api.app(scope, receive, send), 2)
                self.assertNotIn(token.split(".")[0], main_agent._active_sessions)
        self.assertTrue(entered.is_set())
        self.assertTrue(closed.is_set(), "disconnect必须在模型等待期间关闭核心生成器")
        self.assertEqual(sent[0]["status"], 200)
        payload = b"".join(message.get("body", b"") for message in sent).decode()
        self.assertNotIn("event: error", payload)
        self.assertNotIn("event: done", payload)


if __name__ == "__main__":
    unittest.main()
