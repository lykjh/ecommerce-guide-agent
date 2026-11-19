# Agent 服务接口

更新日期：2026-11-08｜HTTP v1.1本地交付C-fix-20261104-2已通过C-1指定离线独立复验；代理/隔离启动器及02宿主受控本地子项已有独立验收记录，真实身份/宿主联调未完成。本次仅更正接入状态，路径、字段、错误与通知协议不变，非生产上线验收。

本文件是浏览器/商城调用 Agent 的接口入口。接口事实依据 [http_api.py](../../agent/http_api.py)；内部调用见[内部工具契约](内部工具契约.md)，商品上游见[商城接口](商城接口.md)。接口版本不代表真实商城身份联调或上线验收已经通过。

## 1. 服务与路径

服务由 FastAPI 提供，OpenAPI 标题为“电商客服 Agent”、版本1.1.0。地址由启动命令的 host/port 决定；页面使用同源相对路径。当前默认本地8000，端口占用时可用8001；不能把其他项目的FastAPI页面当作本服务。启动及排障见[维护手册](../10-操作与维护/维护手册.md)。

| 方法与路径 | 请求 | 成功响应 |
| --- | --- | --- |
| GET `/` | 无 | 200，独立浮窗预览HTML |
| GET `/ui/chat.js`、`/ui/style.css` | 无 | 200，静态资源 |
| GET `/health` | 无 | 200，JSON `{"status":"ok"}`；只验证进程存活 |
| GET `/docs`、`/openapi.json` | 无 | FastAPI自动生成的接口说明/定义 |
| POST `/api/sessions` | 无业务请求体；Origin及商城身份确认，详见第7节 | 201，JSON `{"session_token":"<string>"}` |
| POST `/api/chat` | Origin、Bearer凭证、同域Cookie和JSON消息 | 200，`text/event-stream` |

创建会话不调用模型；不会返回历史列表。订单和物流通过现有聊天端点查询。查询历史、删除历史、商城登录和人工接待没有独立端点。

## 2. 会话凭证

`session_token`为32位十六进制随机会话ID、点号、64位十六进制HMAC-SHA256签名。服务使用环境变量`AGENT_SESSION_SECRET`签名，密钥长度至少32字符；缺失或过短返回503。密钥不返回前端，前端只在`sessionStorage`中保留token。

请求头：`Authorization: Bearer <session_token>`。服务校验格式与签名、查询持久归属并确认商城身份后，才使用随机ID作为内部`thread_id`。客户端不能指定`thread_id`或`userId`替代身份。guest是匿名持有者凭证；user归属还绑定用户ID和登录会话指纹。无归属记录或身份不匹配不得进入图，具体状态码见第7节。

保持同一记忆库、归属库和签名密钥，且当前身份仍匹配，重启后才可继续原上下文。无归属记录的旧匿名token失效，不自动迁移。当前页面刷新不回显历史记录；新会话切换凭证，不承诺物理删除旧历史。更换签名密钥会使旧token失效。

## 3. 聊天输入

```json
{"message":"推荐500元以内的黑色耳机"}
```

仅允许`message:string`。严格字符串，原文长度1..6000，不能只有空白；禁止额外字段。入口不对有效原文trim后改写。缺字段、错误类型、空白、超长、非法JSON或额外字段返回422，不进入图执行。HTTP入口通过Pydantic校验一次后直接调用`stream_reply`。

## 4. SSE输出

UTF-8文本，每帧由空行分隔；`data`必须作为JSON解析。响应包含`Cache-Control: no-cache, no-store`、`X-Accel-Buffering: no`。前端用`fetch`发送POST并读取流，不能直接用只发GET的原生`EventSource`替代。

| event | data字段与类型 | 语义 |
| --- | --- | --- |
| `meta` | `request_id:string` | 本请求追踪编号，首帧 |
| `delta` | `text:string` | 可展示的回复片段，可有多帧 |
| `done` | `request_id:string`、可选 `product_links:array` | 正常流结束，编号与 meta 一致 |
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

商城浮窗可在聊天请求上附 `?product_links=1`。商品推荐、概览、咨询和对比成功结束后，`done.product_links` 最多包含八个本轮已核实条目，按业务节点顺序去重。每项含 `product_id:string` 和 `name:string`，商品 ID 为 1～19 位正十进制字符串。无合格条目、未声明支持、预算停止或请求中止时不下发商品链接。浮窗收到当前请求的 `done` 后，根据 ID 生成同源 `/product/:id` 地址，以新标签页打开。文字帧沿用原格式。

商品咨询/对比允许展示回答模型的可见输出；路由、抽取、信号、摘要和闲聊守卫等`internal`调用始终隐藏。闲聊先整段生成、复核和必要重写，再输出定稿；页面按每20ms最多2个Unicode码点逐步显示。这是审核后的渐进展示，不会提前暴露未审核token，也不会增加模型调用。收起/隐藏页面或减少动态效果时不等待动画；显示完收到的片段后再处理done。

## 5. 错误与取消

开始SSE前的标准业务错误格式：

```json
{"error":{"code":"INVALID_SESSION","message":"请新建会话后重试。"}}
```

| HTTP状态 | code | 触发条件 |
| --- | --- | --- |
| 401 | `INVALID_SESSION` | token缺失、格式/签名错误或归属记录不存在 |
| 422 | `INVALID_REQUEST` | 输入不符合聊天schema；不回显原始输入 |
| 503 | `CONFIGURATION_ERROR` | 会话签名密钥、商城根地址或公开origin配置错误 |

身份新增的403/409/502/503错误及固定文案统一见第7.2节。

开始SSE后的未知框架异常使用`error`事件，code为`AGENT_FAILED`，message为“本轮服务异常，请稍后重试。”，不输出异常正文或堆栈。框架自带未知路径404不属于以上已定义业务错误包；当前前端对不符合客服结构的HTTP错误显示HTTP状态和地址检查提示。

用户停止通过`AbortController`取消fetch。服务监听断开，关闭内部生成器并取消后续调度；取消不是普通Exception，必须传播。取消后已显示内容保留；不保证回滚已提交checkpoint，也不保证撤销供应商已经处理的调用费用。同步工具线程尚未真正退出时继续占用工具名额。

## 6. 并发、日志与接入边界

- 相同内部会话同一时刻只接纳一轮，其他请求收到忙提示，不调用模型、不写入图；不排队或自动重发。
- 不同聊天会话可并行；request_id、队列、预算、callback和持久状态各自独立。
- 当前支持单worker、单事件循环。同库checkpoint IO锁只协调这一范围；不保证多进程/多事件循环共享库安全。
- HTTP的request_id贯通图、模型用量和预算日志。token、Authorization、原始输入/输出不得进入普通日志。
- 当前无CORS放行；身份、归属、代理和宿主页面已有本地交付与受控验证，真实A/B及实际运行环境联调未完成。仍无用户/IP限流及公网部署验收承诺。
- 修改路径、字段、事件或错误语义时同步修改前端、OpenAPI和相关测试；不能为兼容旧测试改变正确业务设计。

对应离线测试：[HTTP契约](../../tests/test_http_api.py)、[页面统一入口](../../tests/test_http_ui.py)、[页面事件用例](../../tests/http_ui.test.mjs)。测试是否通过及其日期以[验收报告](../09-验收与交付/验收报告.md)为准。

## 7. 身份阶段契约HTTP v1.1（本地实现已交付，验收未完成）

本节约束当前身份实现；第1—6节描述通用路径与SSE。实现原理、存储与时序见[身份与会话归属实现设计](D:/desktop/PythonProject/docs/04-工程/身份与会话归属实现设计.md)。本节是身份路径、字段、错误码与宿主通知的唯一接口来源；/assistant/本地代理与宿主通知已有受控交付，真实环境仍待验证。既有阶段结论以[施工进度](D:/desktop/PythonProject/docs/05-施工进度/施工进度.md)和[02任务卡](D:/desktop/PythonProject/docs/05-施工进度/任务单/V1剩余工作五人分工/02-商城宿主客服窗口.md#2026-11-05-交付核对)为准，本次没有新增业务执行结果。

### 7.1 路径、请求与响应

默认同域公开前缀/assistant/，代理剥离前缀；服务内部路径不改。root_path和代理前缀必须一起配置；本地直连调试可使用根路径。公共origin来自AGENT_PUBLIC_ORIGIN，不把localhost写成部署常量。

| 对外路径 | 方法/请求 | 成功响应 |
| --- | --- | --- |
| /assistant/api/sessions | POST，无业务请求体；必须匹配的Origin；浏览器自动携同域Cookie；不需Bearer | 201 application/json：session_token:string；只在归属已落库后返回 |
| /assistant/api/chat | POST；Origin、Authorization: Bearer <session_token>；application/json且仅message:string，原1..6000字符/非空白约束不变；同域Cookie | 200 text/event-stream；沿用meta/delta/done/error及字段类型 |
| /assistant/ | GET | 独立客服页面 |
| /assistant/ui/* | GET | 静态资源 |
| /assistant/health | GET | 原存活响应，不表示身份依赖就绪 |

session_token格式仍为32位随机ID、点号和64位签名。不接受userId、role、thread_id或用户指定商城URL作为身份。Agent 将请求中的 `MALLAUTH` 转发给配置商城的 `GET /users/me` 确认身份；Cookie 缺失为 guest。身份确认与归属比较在开始SSE之前完成，不发送“先进入图再鉴权”的meta。

接口不返回用户ID、原始Cookie、登录指纹或新的账号token。首次HTTP v1.1启用后，无归属记录的旧token失效；旧历史不删除也不自动迁移。guest归属与user归属不可互换，user还必须匹配本次登录会话指纹；签名有效不再等于有权读取该聊天。

### 本地商城前端映射补充（2026-11-08，本地受控通过、真实联验待执行）

宿主D:/desktop/vue-mall/mall的Vite配置端口为5173，商城请求使用/api前缀；例如浏览器/api/users/me转到商城/users/me。客服沿用同一http://localhost:5173下的/assistant/前缀，客服/api/sessions和/api/chat在浏览器分别为/assistant/api/sessions及/assistant/api/chat，不能与商城/api代理混用。5173是既有开发配置，真实批次仍须登记实际origin与运行目标，不表示该地址当前已有可用服务。

Agent服务端MALL_BASE_URL仍是商城真实根地址，查询/users/me不带浏览器/api前缀。AGENT_PUBLIC_ORIGIN应与统一浏览器origin一致；代理target是服务间地址，不能替代这个origin。商城Axios封装处理code/data包络，客服仍由自身fetch处理JSON/SSE。

本地代理、示例配置和隔离启动器已按[C-2任务](D:/desktop/PythonProject/docs/05-施工进度/任务单/C-2-真实商城身份与会话隔离联调.md)完成准备子项；不重复修改已通过的配置。真实联调按该卡第11节登记同一候选版、专用A/B、隔离双库、操作范围及经确认的数量预算，材料/授权未齐不得启动。Nginx不在本次本地联调写入范围，本地子项不表示生产部署完成。

### 7.2 HTTP错误及前端处理

结构仍为error对象，内含code:string、message:string。下表message为固定文案，不拼上游异常或账号详情。HTTP错误发生在SSE前；无错误后自动重试或自动重发用户消息。

| HTTP | code | 触发条件 | message / 页面动作 |
| --- | --- | --- | --- |
| 401 | INVALID_SESSION | token缺失、格式/签名错误或归属记录不存在 | 请新建会话后重试。 / 清空旧会话，下一次发送创建新会话 |
| 409 | SESSION_IDENTITY_CHANGED | 已确认身份与归属类型、用户ID或登录指纹不符 | 登录身份已变化，请在新会话中重新发送。 / 清空旧会话与输入，不迁移、不自动重发 |
| 403 | ORIGIN_NOT_ALLOWED | Origin缺失、null或不匹配配置 | 请求来源不受支持。 / 不发起聊天 |
| 422 | INVALID_REQUEST | 仍按第3节消息schema | 请输入 1 至 6000 字符的非空文本。 / 提示修正 |
| 502 | IDENTITY_BAD_RESPONSE | 身份上游包络/字段/状态不符合设计第3节约定 | 登录状态响应异常，请稍后重试。 / 保留当前会话 |
| 503 | IDENTITY_UNAVAILABLE | 上游403/429/5xx/3xx、网络失败或5秒截止 | 暂时无法确认登录状态，请稍后重试。 / 保留当前会话，不能降级游客继续 |
| 503 | IDENTITY_BUSY | 4个身份HTTP槽位均未释放 | 登录状态校验繁忙，请稍后重试。 / 不排队、不自动重试 |
| 503 | SESSION_STORE_UNAVAILABLE | 归属存储无法读写/超时 | 会话服务暂时不可用，请稍后重试。 / 不返回新token，不进入图 |
| 503 | CONFIGURATION_ERROR | 签名密钥、商城目标或公开origin配置缺失/非法 | 客服服务配置未完成。 / 不静默转匿名模式 |

上游正常401表示未登录，在身份解析边界得到guest；如果请求持有user归属聊天token，则返回409。不能把上游503映射成guest或“没有订单”。开始SSE后的未知异常继续使用AGENT_FAILED；本方案不新增业务成功事件，也不修改预算/忙提示的现行语义。

### 7.3 宿主身份变化通知

通知内容：type:string固定mall:identity-changed，version:integer固定1；无其它必需字段，不承载账号、Cookie、聊天token或历史。宿主成功登录/退出/换号或确认登录失效后，调用iframe.contentWindow.postMessage通知，targetOrigin精确为当前origin。客服只接受同origin且source为window.parent的消息，校验一次后执行重置。

同域多标签页频道固定xiaonuan.identity.v1，数据结构相同。客服收到父页面通知后广播一次；收到频道消息只重置，不转发。宿主页即使没有打开客服，也应发布频道通知。通知仅取消旧流、清空旧上下文并使下次请求重新认证，不能替代/users/me认证。首次接入宿主页须按此契约验收；未接入时仅能证明客服侧协议能力。

前端以页面generation隔离异步结果，身份变化后任何旧请求的token响应、delta、错误或finally都不得改写新页面。服务器只在每次请求入口确认身份，没有商城实时撤销推送；已通过认证的在途公开商品回答可能持续至原截止时间，宿主通知到达则页面立即停止并中止连接。私有订单查询另在首次输出订单事实前复核本轮登录，见第 8 节。

## 8. 本人订单与物流

从聊天发送「我的订单」「订单 123」「订单 123 的物流」。登录买家可在当前会话继续用「第二个」「下一页」「这单的物流」；「返回导购」离开订单模式。列表每页最多 5 条，游标由服务端保存，与当前已验证登录绑定。商城逐次验证订单和游标归属，游客收到登录引导。

订单分支以确定性规则调用本人列表、详情和包裹三条 GET，回复仍用 `delta` / `done`。只显示所需订单字段及脱敏运单，查询内容不进入模型或聊天 checkpoint。非本人和不存在的订单使用同一提示。

若开始 SSE 后商城凭证失效，服务发送 `error`，`code=ORDER_AUTH_EXPIRED`。本轮不输出订单数据，也不发送 `done`。浮窗清除旧会话和旧订单展示，通知同源宿主确认登录状态，由用户重新发送请求。接入和核验记录见[本人订单与物流接入设计](../04-工程/本人订单与物流接入设计.md)。
