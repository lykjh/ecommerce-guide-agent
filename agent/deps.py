"""共享配置与按需创建的模型客户端。"""
import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from agent.utils.config_tool import get_app_config

APP_CFG = get_app_config()
DB_PATH = APP_CFG.abs_path("path.db")
INTERNAL_TAG = "internal"

def build_client(model_key, temperature, *, max_tokens=1200, **kwargs):
    return init_chat_model(
        model=APP_CFG.get(model_key),
        model_provider=APP_CFG.get("llm.model_provider", "openai"),
        api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY")),
        base_url=os.getenv(APP_CFG.get("llm.base_url_env")),
        temperature=temperature, max_tokens=max_tokens, max_retries=0, timeout=10,
        extra_body=APP_CFG.get("llm.extra_body", {}), **kwargs,
    )

@lru_cache(maxsize=1)
def get_client():
    return build_client("llm.model", APP_CFG.get("llm.temperature", 0.7),
                        max_tokens=min(APP_CFG.get("llm.max_tokens", 1024), 1200))

@lru_cache(maxsize=1)
def get_extract_client():
    return build_client("llm.router_model", 0,
                        max_tokens=min(APP_CFG.get("llm.max_tokens", 1024), 1200))
