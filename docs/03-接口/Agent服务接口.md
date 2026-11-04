# Agent 服务接口

更新日期：2026-11-03｜契约版本：HTTP v1.0，页面展示行为更新至2026-11-03｜状态：现行实现说明，非生产上线验收。

本文件是浏览器/商城调用 Agent 的接口入口。接口事实依据 [http_api.py](../../agent/http_api.py)；内部调用见[内部工具契约](内部工具契约.md)，商品上游见[商城接口](商城接口.md)。接口版本不代表商城身份已经接通。

## 1. 服务与路径

服务由 FastAPI 提供，OpenAPI 标题为“电商客服 Agent”、版本1.0.0。地址由启动命令的 host/port 决定；页面使用同源相对路径。当前默认本地8000，端口占用时可用8001；不能把其他项目的FastAPI页面当作本服务。启动及排障见[维护手册](../10-操作与维护/维护手册.md)。

| 方法与路径 | 请求 | 成功响应 |
| --- | --- | --- |
| GET `/` | 无 | 200，独立浮窗预览HTML |
| GET `/ui/chat.js`、`/ui/style.css` | 无 | 200，静态资源 |
| GET `/health` | 无 | 200，JSON `{"status":"ok"}`；只验证进程存活 |
| GET `/docs`、`/openapi.json` | 无 | FastAPI自动生成的接口说明/定义 |
| POST `/api/sessions` | 无业务请求体，无商城身份 | 201，JSON `{"session_token":"<string>"}` |
| POST `/api/chat` | Bearer凭证和JSON消息 | 200，`text/event-stream` |

创建会话不调用模型；不会返回历史列表。没有查询历史、删除历史、商城登录、订单或人工接待HTTP端点。

## 2. 会话凭证

`session_token`为32位十六进制随机会话ID、点号、64位十六进制HMAC-SHA256签名。服务使用环境变量`AGENT_SESSION_SECRET`签名，密钥长度至少32字符；缺失或过短返回503。密钥不返回前端，前端只在`sessionStorage`中保留token。

请求头：`Authorization: Bearer <session_token>`。服务校验格式与签名，再解出内部`thread_id`；客户端不能提交任意`thread_id`或`userId`替代凭证。token是匿名持有者凭证，没有商城账号绑定和到期字段；持有有效token即可访问该会话，不可把匿名隔离测试当作跨账号权限证明。

同一数据库和同一签名密钥可在服务重启后继续原上下文。当前页面刷新不回显历史记录；新会话切换凭证，不承诺物理删除旧历史。更换签名密钥会使旧token失效。

## 3. 聊天输入

```json
{"message":"推荐500元以内的黑色耳机"}
```

仅允许`message:string`。严格字符串，原文长度1..6000，不能只有空白；禁止额外字段。入口不对有效原文trim后改写。缺字段、错误类型、空白、超长、非法JSON或额外字段返回422，不进入图执行。HTTP入口通过Pydantic校验一次后直接调用`stream_reply`。

## 4. SSE输出

UTF-8文本，每帧由空行分隔；`data`必须作为JSON解析。响应包含`Cache-Control: no-cache`、`X-Accel-Buffering: no`。前端用`fetch`发送POST并读取流，不能直接用只发GET的原生`EventSource`替代。

| event | data字段与类型 | 语义 |
| --- | --- | --- |
| `meta` | `request_id:string` | 本请求追踪编号，首帧 |
| `delta` | `text:string` | 可展示的回复片段，可有多帧 |
| `done` | `request_id:string` | 正常流结束，编号与meta一致 |
| `error` | `code:string`、`message:string`、`request_id:string` | 开始SSE后的服务异常；随后关流，不再发done |

```text
event: meta
data: {"request_id":"example-request"}

event: delta
data: {"text":"你好"}

event: done
data: {"request_id":"example-request"}

```

`done`只表示输出流结束，不代表订单查询、转人工或全部业务成功。同会话忙和预算停止目前是普通文字delta后done，不是409/429，也没有额外业务成功字段。

商品咨询/对比允许展示回答模型的可见输出；路由、抽取、信号、摘要和闲聊守卫等`internal`调用始终隐藏。闲聊先整段生成、复核和必要重写，再输出定稿；页面按每20ms最多2个Unicode码点逐步显示。这是审核后的渐进展示，不会提前暴露未审核token，也不会增加模型调用。收起/隐藏页面或减少动态效果时不等待动画；显示完收到的片段后再处理done。

## 5. 错误与取消

开始SSE前的标准业务错误格式：

```json
{"error":{"code":"INVALID_SESSION","message":"请新建会话后重试。"}}
```

| HTTP状态 | code | 触发条件 |
| --- | --- | --- |
| 401 | `INVALID_SESSION` | 缺失、格式非法或签名不匹配的token |
| 422 | `INVALID_REQUEST` | 输入不符合聊天schema；不回显原始输入 |
| 503 | `CONFIGURATION_ERROR` | 会话签名密钥未正确配置 |

开始SSE后的未知框架异常使用`error`事件，code为`AGENT_FAILED`，message为“本轮服务异常，请稍后重试。”，不输出异常正文或堆栈。框架自带未知路径404不属于以上已定义业务错误包；当前前端对不符合客服结构的HTTP错误显示HTTP状态和地址检查提示。

用户停止通过`AbortController`取消fetch。服务监听断开，关闭内部生成器并取消后续调度；取消不是普通Exception，必须传播。取消后已显示内容保留；不保证回滚已提交checkpoint，也不保证撤销供应商已经处理的调用费用。同步工具线程尚未真正退出时继续占用工具名额。

## 6. 并发、日志与接入边界

- 相同内部会话同一时刻只接纳一轮，其他请求收到忙提示，不调用模型、不写入图；不排队或自动重发。
- 不同匿名会话可并行；request_id、队列、预算、callback和持久状态各自独立。
- 当前支持单worker、单事件循环。同库checkpoint IO锁只协调这一范围；不保证多进程/多事件循环共享库安全。
- HTTP的request_id贯通图、模型用量和预算日志。token、Authorization、原始输入/输出不得进入普通日志。
- 当前无CORS放行，无商城身份、用户/IP限流和公网部署承诺。实际宿主页、可信身份传播、退出换号和网关规则尚待冻结。
- 修改路径、字段、事件或错误语义时同步修改前端、OpenAPI和相关测试；不能为兼容旧测试改变正确业务设计。

对应离线测试：[HTTP契约](../../tests/test_http_api.py)、[页面统一入口](../../tests/test_http_ui.py)、[页面事件用例](../../tests/http_ui.test.mjs)。测试是否通过及其日期以[验收报告](../09-验收与交付/验收报告.md)为准。
