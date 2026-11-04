# 电商客服 Agent

当前Agent已有商品接口适配；用户确认的现行后端为`D:/desktop/IdeaProjects/shiwushop`，商品/分类源码契约已核对，运行服务和真实样例待验证，接入状态见[商城接口](docs/03-接口/商城接口.md)。Agent技术栈：FastAPI + LangGraph + 可收起聊天窗口。支持商品咨询/推荐/对比/概览、多轮记忆、预算与超时；政策、本人订单/物流和真实人工尚未接入。完整V1未通过上线验收。

## 启动与使用

在项目根目录、配置好本机环境后运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn agent.http_api:app --host 127.0.0.1 --port 8001 --workers 1
```

打开 http://127.0.0.1:8001/ 。端口可选；最近8000被其他项目占用。当前验证范围单worker、单事件循环。详细配置、模型Key和签名密钥要求见[维护手册](docs/10-操作与维护/维护手册.md)，不要覆盖既有.env或提交密钥。

## 文档与测试

[完整文档导航](docs/README.md) · [用户操作](docs/10-操作与维护/用户操作手册.md) · [当前施工进度](docs/05-施工进度/施工进度.md) · [验收结果](docs/09-验收与交付/验收报告.md)

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p "test_*.py"
node tests/http_ui.test.mjs
```

Python总回归包含Node.js执行的前端用例，需安装Node.js；详细测试条件见[测试计划](docs/08-测试/测试计划.md)。live_开头的测试须显式执行，会涉及真实模型/网络和费用，不能在文档迁移时自动运行。

业务代码在agent，配置、测试、文档分别在根config、tests、docs。历史商品文本rag_text不是已验收的政策知识库。文档入口见[文档导航](docs/README.md)。
