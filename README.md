# ecommerce-guide-agent

基于 LangGraph + RAG 的电商导购 Agent。

## 能力

- **意图路由**：闲聊 / 商品咨询自动分流
- **商品问答**：Chroma 向量库 + DashScope Embedding，答案来自知识库而非模型编造
- **知识库增量同步**：按文件内容 md5 去重，重复运行只处理新增和改动的文件
- **分层记忆**：最近窗口 + 会话摘要，token 不随对话轮数线性膨胀
- **流式输出**：Streamlit 界面逐字返回

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 OPENAI_BASE_URL 和 OPENAI_API_KEY
python -m rag.build_db      # 构建向量库
streamlit run web.py
```

## 注意

模型地址与 API Key 全部走环境变量，不写入配置文件。`rag_text/` 为商品知识原始文本。
