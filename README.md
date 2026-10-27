# ecommerce-guide-agent

基于 LangGraph + RAG 的电商导购 Agent。

## 能力

- **意图路由**：闲聊 / 商品咨询自动分流（qwen-flash 轻量分类）
- **商品问答**：Chroma 向量库 + DashScope Embedding，答案来自知识库而非模型编造
- **知识库增量同步**：按文件内容 md5 去重，重复运行只处理新增和改动的文件
- **分层记忆**：L1 最近窗口 + L2 会话摘要，发给模型的消息条数不随对话轮数线性膨胀
- **流式输出**：Streamlit 界面逐字返回

## 目录结构

```
agent/        图编排、意图路由、记忆管理
rag/          知识库构建（build_db）与检索（retriever）
utils/        配置加载、日志、路径
config/       app.yml（模型与路径）、chroma.yml（向量库与切分）
rag_text/     商品知识原始文本
web.py        Streamlit 聊天界面
```

## 快速开始

```bash
pip install -r requirements.txt      # 依赖清单需自行生成
cp .env.example .env                 # 填入 OPENAI_BASE_URL 与 OPENAI_API_KEY
python -m rag.build_db               # 构建 / 增量同步向量库
streamlit run web.py                 # 启动界面
```

> 必须用 `python -m rag.build_db` 运行，直接 `python rag/build_db.py` 会因包内导入失败。

## 配置说明

模型地址与 API Key 全部走环境变量，配置文件里只存变量名，避免密钥进仓库：

| 环境变量 | 用途 |
|---|---|
| `OPENAI_BASE_URL` | 兼容 OpenAI 协议的服务地址 |
| `OPENAI_API_KEY` | DashScope / 百炼 API Key |

向量库相关参数（集合名、切分大小、召回条数、距离阈值）在 `config/chroma.yml`。

## 说明

`rag_text/` 内为示例商品数据。模型地址若使用云厂商的专属网关（含租户标识），请勿写入配置文件。
