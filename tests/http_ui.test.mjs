import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import test from "node:test";

// 只去掉模块导出标记，在最小DOM替身中执行原页面事件处理器。
const source = readFileSync(new URL("../agent/http_ui/chat.js", import.meta.url), "utf8")
  .replace("export async function*", "async function*");

function page(responses, savedToken = null, { paced = false, reducedMotion = false } = {}) {
  const elements = new Map();
  const storage = new Map(savedToken ? [["xiaonuan-session", savedToken]] : []);
  const requests = [];
  const ticks = [];
  let onTick;
  const firstTick = new Promise(resolve => { onTick = resolve; });
  function element() {
    return { value: "", textContent: "", hidden: false, children: [], handlers: {},
      addEventListener(name, handler) { this.handlers[name] = handler; },
      setAttribute() {}, focus() {},
      append(...children) { this.children.push(...children); },
      replaceChildren() { this.children = []; },
    };
  }
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, element());
      return elements.get(selector);
    },
    querySelectorAll() { return []; }, createElement: element,
  };
  runInNewContext(source, { document, TextDecoder, AbortController,
    window: { matchMedia: () => ({ matches: reducedMotion }) },
    setTimeout: callback => {
      if (paced) { ticks.push(callback); onTick(); }
      else callback();
    },
    sessionStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    fetch: async (url, options) => {
      requests.push({ url, options });
      assert.ok(responses.length, "出现未预期的额外请求");
      return responses.shift();
    },
  });
  return { elements, storage, requests, ticks, firstTick,
    async submit() {
      elements.get("#message").value = "你好";
      await elements.get("#chat-form").handlers.submit({ preventDefault() {} });
    },
  };
}
const json = (body, status) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json" },
});

for (const [name, token] of [["创建会话", null], ["已有会话聊天", "test-session"]]) {
  test(`${name}遇到其他服务404时显示地址提示`, async () => {
    const ui = page([json({ detail: "Not Found" }, 404)], token);
    await ui.submit();
    assert.match(ui.elements.get("#status").textContent, /HTTP 404/);
    assert.match(ui.elements.get("#status").textContent, /客服服务地址/);
    assert.doesNotMatch(ui.elements.get("#status").textContent, /Cannot read/);
    assert.equal(ui.requests.length, 1);
    assert.equal(ui.elements.get("#send").disabled, false);
  });
}

test("客服标准错误保留服务端提示", async () => {
  const ui = page([json({ error: { code: "CONFIGURATION_ERROR", message: "会话密钥尚未配置。" } }, 503)]);
  await ui.submit();
  assert.match(ui.elements.get("#status").textContent, /会话密钥尚未配置/);
});

test("401仍清除旧凭证且不自动重发", async () => {
  const ui = page([json({ error: { code: "INVALID_SESSION", message: "请新建会话后重试。" } }, 401)], "old-token");
  await ui.submit();
  assert.equal(ui.storage.has("xiaonuan-session"), false);
  assert.equal(ui.requests.length, 1);
  assert.match(ui.elements.get("#status").textContent, /请新建会话/);
});

test("正常创建会话及SSE回复不受影响", async () => {
  const frames = 'event: meta\ndata: {"request_id":"test-request"}\n\n'
    + 'event: delta\ndata: {"text":"你好，我是小暖。"}\n\n'
    + 'event: done\ndata: {"request_id":"test-request"}\n\n';
  const ui = page([json({ session_token: "new-token" }, 201), new Response(frames)]);
  await ui.submit();
  assert.equal(ui.elements.get("#status").textContent, "本轮回复结束");
  assert.equal(ui.elements.get("#messages").children[1].children[1].textContent, "你好，我是小暖。");
  assert.equal(ui.storage.get("xiaonuan-session"), "new-token");
  assert.equal(ui.requests[1].options.headers.Authorization, "Bearer new-token");
});

const answerFrames = text => 'event: meta\ndata: {"request_id":"typing-test"}\n\n'
  + `event: delta\ndata: ${JSON.stringify({ text })}\n\n`
  + 'event: done\ndata: {"request_id":"typing-test"}\n\n';

test("合并到达的回复也逐步显示，停止保留已显示部分", async () => {
  const ui = page([new Response(answerFrames("你好🙂这是审核后的完整回复"))], "test-token", { paced: true });
  const submitted = ui.submit();
  const stage = await Promise.race([ui.firstTick.then(() => "typing"), submitted.then(() => "done")]);
  assert.equal(stage, "typing", "整段SSE同批到达时不应一次显示全部文字");
  const reply = ui.elements.get("#messages").children[1].children[1];
  assert.equal(reply.textContent, "你好");
  assert.equal(ui.elements.get("#send").disabled, true);
  ui.elements.get("#stop").handlers.click();
  ui.ticks.shift()();
  await submitted;
  assert.match(reply.textContent, /你好/);
  assert.doesNotMatch(reply.textContent, /完整回复/);
  assert.match(ui.elements.get("#status").textContent, /已停止/);
  assert.equal(ui.elements.get("#send").disabled, false);
});

test("减少动态效果时直接展示并正确保留emoji", async () => {
  const text = "你好🙂欢迎";
  const ui = page([new Response(answerFrames(text))], "test-token", { paced: true, reducedMotion: true });
  await ui.submit();
  assert.equal(ui.ticks.length, 0);
  assert.equal(ui.elements.get("#messages").children[1].children[1].textContent, text);
});

test("窗口收起时不等待打字动画", async () => {
  const ui = page([new Response(answerFrames("完整回复"))], "test-token", { paced: true });
  ui.elements.get("#minimize").handlers.click();
  await ui.submit();
  assert.equal(ui.ticks.length, 0);
  assert.equal(ui.elements.get("#messages").children[1].children[1].textContent, "完整回复");
});
