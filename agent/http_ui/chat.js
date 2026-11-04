// 流读取是协议边界；TextDecoder 保留被网络切开的 UTF-8 字节。
export async function* readEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let separator;
      while ((separator = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator[0].length);
        let event = "message";
        const data = [];
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith("event:")) event = line.slice(6).trimStart();
          if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (data.length) yield { event, data: JSON.parse(data.join("\n")) };
      }
      if (done) {
        if (buffer.trim()) throw new Error("回复数据不完整，请重新发送。");
        return;
      }
    }
  } finally {
    try {
      await reader.cancel();
    } finally {
      reader.releaseLock();
    }
  }
}

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const stopButton = document.querySelector("#stop");
const newButton = document.querySelector("#new-session");
const status = document.querySelector("#status");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const panel = document.querySelector("#chat-panel");
const launcher = document.querySelector("#launcher");
const launcherLabel = document.querySelector("#launcher-label");
const launcherNotice = document.querySelector("#launcher-notice");
const minimize = document.querySelector("#minimize");
const chatScroll = document.querySelector("#chat-scroll");
const suggestions = document.querySelectorAll("[data-prompt]");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const tokenKey = "xiaonuan-session";
let token = sessionStorage.getItem(tokenKey);
let controller = null;
let unread = false;

function updateLauncher() {
  launcher.setAttribute("aria-expanded", String(!panel.hidden));
  launcher.setAttribute("aria-label", panel.hidden ? "展开客服窗口" : "收起客服窗口");
  launcherLabel.textContent = panel.hidden ? "小暖客服" : "收起小暖";
  launcherNotice.hidden = !panel.hidden || (!controller && !unread);
  launcherNotice.textContent = controller ? "正在回复…" : "有新回复";
}

function setPanelOpen(open) {
  panel.hidden = !open;
  if (open) {
    unread = false;
    chatScroll.scrollTop = chatScroll.scrollHeight;
    (input.disabled ? minimize : input).focus();
  } else {
    launcher.focus();
  }
  updateLauncher();
}

launcher.addEventListener("click", () => setPanelOpen(panel.hidden));
minimize.addEventListener("click", () => setPanelOpen(false));
panel.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    setPanelOpen(false);
  }
});
for (const button of suggestions) {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
}
updateLauncher();

function setBusy(busy) {
  sendButton.disabled = busy;
  newButton.disabled = busy;
  input.disabled = busy;
  stopButton.disabled = !busy;
  for (const button of suggestions) button.disabled = busy;
  updateLauncher();
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "小暖";
  const content = document.createElement("p");
  content.className = "message-content";
  content.textContent = text;
  article.append(label, content);
  messages.append(article);
  welcome.hidden = true;
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return content;
}

function responseError(result, statusCode) {
  // HTTP错误在接入边界解析；其他服务的404不一定使用客服的错误结构。
  const message = result?.error?.message;
  return new Error(typeof message === "string" ? message
    : `客服接口返回异常（HTTP ${statusCode}），请确认打开的是客服服务地址。`);
}

async function createSession(signal) {
  const response = await fetch("/api/sessions", { method: "POST", signal });
  const result = await response.json();
  if (!response.ok) throw responseError(result, response.status);
  token = result.session_token;
  sessionStorage.setItem(tokenKey, token);
}

newButton.addEventListener("click", async () => {
  controller = new AbortController();
  setBusy(true);
  status.textContent = "正在创建新会话…";
  try {
    await createSession(controller.signal);
    messages.replaceChildren();
    welcome.hidden = false;
    unread = false;
    status.textContent = "新会话已开始";
  } catch (error) {
    status.textContent = error.name === "AbortError" ? "已停止" : `创建失败：${error.message}`;
  } finally {
    controller = null;
    setBusy(false);
    if (!panel.hidden) input.focus();
  }
});

stopButton.addEventListener("click", () => controller.abort());
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value;
  if (!message.trim()) {
    status.textContent = "请输入问题后再发送。";
    input.focus();
    return;
  }
  controller = new AbortController();
  setBusy(true);
  addMessage("user", message);
  const reply = addMessage("assistant", "");
  input.value = "";
  status.textContent = "正在连接客服…";
  let requestId = "";
  let finished = false;
  try {
    if (!token) await createSession(controller.signal);
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const result = await response.json();
      if (response.status === 401) {
        token = null;
        sessionStorage.removeItem(tokenKey);
      }
      throw responseError(result, response.status);
    }
    for await (const { event, data } of readEvents(response.body)) {
      controller.signal.throwIfAborted();
      switch (event) {
        case "meta":
          requestId = data.request_id;
          status.textContent = "正在准备回复…";
          break;
        case "delta":
          status.textContent = "正在回复…";
          // 审核后的分片可能同批到达；按Unicode码点展示，避免切断代理对。
          const characters = Array.from(data.text);
          for (let i = 0; i < characters.length; i += 2) {
            controller.signal.throwIfAborted();
            reply.textContent += characters.slice(i, i + 2).join("");
            chatScroll.scrollTop = chatScroll.scrollHeight;
            if (!document.hidden && !panel.hidden && !reducedMotion.matches) {
              await new Promise(resolve => setTimeout(resolve, 20));
            }
          }
          break;
        case "error":
          throw new Error(data.message);
        case "done":
          finished = true;
          status.textContent = "本轮回复结束";
          break;
        default:
          throw new Error("收到未识别的回复事件。");
      }
      if (finished) break;
    }
    if (!finished) throw new Error("连接已中断，回复可能不完整，请重新发送。");
  } catch (error) {
    const notice = error.name === "AbortError" ? "已停止生成，已收到的内容保留。" : `本轮未完成：${error.message}`;
    status.textContent = notice;
    reply.textContent += `${reply.textContent ? "\n\n" : ""}${notice}${requestId ? `\n请求编号：${requestId}` : ""}`;
  } finally {
    controller = null;
    unread = panel.hidden;
    setBusy(false);
    if (!panel.hidden) input.focus();
  }
});
