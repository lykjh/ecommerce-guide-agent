"""只供H的跨进程恢复测试调用，无在线依赖。"""
import asyncio
import json
import os
from pathlib import Path
import sys
import httpx

from agent import http_api
from tests.v1.h.offline_runtime import offline_runtime, snapshot, decode_events


async def run():
    path, input_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(input_path.read_text(encoding="utf-8"))
    with offline_runtime(path):
        before = await snapshot(path, request["session_id"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http_api.app), base_url="http://offline") as client:
            response = await client.post("/api/chat", headers={"Authorization": "Bearer " + request["token"]}, json={"message": "商品SESSION_A 重启后继续"})
        events = decode_events(response.text)
        after = await snapshot(path, request["session_id"])
    output_path.write_text(json.dumps({"pid": os.getpid(), "summary_before": before["summary"], "summary_after": after["summary"], "page_before": before["product_page"], "page_after": after["product_page"], "events": events}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run())
