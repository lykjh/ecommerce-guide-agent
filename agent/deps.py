"""共享依赖：配置、模型客户端、检索器、常量。

为什么单独一个文件：
节点模块（agent/nodes/*.py）需要用到模型客户端和检索器，
但这些东西不能放在 main_agent.py 里——main_agent 要 import 节点，
节点再 import main_agent 就成循环导入了。抽出来给两边共用即可。

⚠️ 注意：这里是「搬运」不是「懒加载」，import 本模块依然会立刻建立连接
（Chroma + 两个大模型客户端）。要做成真懒加载得改成工厂函数，那是另一件事。
"""

import os

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

from rag.retriever import get_retriever
from utils.config_tool import get_app_config, get_rag_config

#应用配置（config/app.yml）
APP_CFG = get_app_config()
#RAG 配置：距离阈值等（config/chroma.yml）
RAG_CFG = get_rag_config()

#SQLite 记忆库绝对路径
DB_PATH = APP_CFG.abs_path("path.db")

#模型服务地址。
#⚠️ 地址不写在 yml 里（专属网关带租户标识，仓库公开会泄露），
#只从环境变量读：yml 里配的是变量名 llm.base_url_env，真实值放 .env
BASE_URL = os.getenv(APP_CFG.get("llm.base_url_env"))


def _build_client(model_key: str, temperature: float):
    """按配置构造一个模型客户端"""
    return init_chat_model(
        model=APP_CFG.get(model_key),
        model_provider=APP_CFG.get("llm.model_provider", "openai"),
        api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY")),
        base_url=BASE_URL,
        temperature=temperature,
        max_tokens=APP_CFG.get("llm.max_tokens", 1024),
    )


#回答用模型：需要一点创造性
client = _build_client("llm.model", APP_CFG.get("llm.temperature", 0.7))

#参数抽取专用模型：抽取是「结构化提取」不是「自由创作」，必须用低温度 + 快模型。
#用上面那个 temperature=0.7 的模型做抽取，既慢又抽不稳（同样的话两次结果可能不同）。
extract_client = _build_client("llm.router_model", 0)

#检索器（向量库连接）。用 rag 层的单例，避免和 tool/ 重复建连接
retriever = get_retriever()

#所有「内部调用」统一打这个 tag，流式输出时靠它把抽取过程挡在界面外。
#（stream_mode="messages" 捕获的是节点内每一次 LLM 调用，不是每个节点一次）
INTERNAL_TAG = "internal"
