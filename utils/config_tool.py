"""统一配置加载器：把 config/*.yml 读进内存。

设计要点：
1. 配置目录固定为 <项目根>/config，项目根由 utils/path_tool.get_pro_root() 推导（基于 __file__，不受工作目录影响）。
2. 加载结果带缓存，一个进程里同一个 yml 只读一次磁盘。
3. 对外只暴露 Config 对象，支持 "a.b.c" 点号取值，避免调用方到处写 data["llm"]["base_url"] 这种硬编码层级。
"""

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

from utils.path_tool import get_pro_root

#配置目录名（相对项目根目录）
CONFIG_DIR_NAME = "config"


def get_config_dir() -> str:
    """config 目录的绝对路径"""
    return os.path.join(get_pro_root(), CONFIG_DIR_NAME)


@lru_cache(maxsize=None)
def load_yaml(file_name: str) -> Dict[str, Any]:
    """加载 config 下的 yml 文件并返回字典（带缓存）。

    :param file_name: 文件名，可带可不带 .yml / .yaml 后缀，例如 "app" 或 "app.yml"
    :return: 配置字典
    """
    if not file_name.endswith((".yml", ".yaml")):
        file_name += ".yml"

    path = os.path.join(get_config_dir(), file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是键值映射（yml 里别写成 key=value 形式）: {path}")
    return data


class Config:
    """配置字典的薄封装：点号取值 + 相对路径自动转绝对路径"""

    def __init__(self, data: Dict[str, Any], root: str = None):
        self._data = data
        #相对路径按哪个目录展开，默认项目根目录
        self.root = root or get_pro_root()

    def get(self, key: str, default: Any = None) -> Any:
        """按 "a.b.c" 取值，取不到返回 default"""
        cur = self._data
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def require(self, key: str) -> Any:
        """按 "a.b.c" 取值，取不到直接报错（适合必填配置）"""
        value = self.get(key)
        if value is None:
            raise KeyError(f"缺少必填配置项: {key}（配置文件目录: {get_config_dir()}）")
        return value

    def abs_path(self, key: str, default: str = "") -> str:
        """读路径类配置：相对路径按项目根目录展开；本来就是绝对路径则原样返回"""
        value = self.get(key, default)
        if not value:
            return value
        if os.path.isabs(value):
            return value
        return os.path.join(self.root, value)

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config({self._data})"


@lru_cache(maxsize=None)
def get_app_config() -> Config:
    """应用全局配置：config/app.yml（路径、LLM 连接参数等）"""
    return Config(load_yaml("app.yml"))


@lru_cache(maxsize=None)
def get_rag_config() -> Config:
    """RAG 配置：config/chroma.yml（向量库、切分参数等）"""
    return Config(load_yaml("chroma.yml"))


if __name__ == "__main__":
    #自检：确认 yml 真的能被加载进来
    print("项目根目录:", get_pro_root())
    print("配置目录  :", get_config_dir())
    app_cfg = get_app_config()
    rag_cfg = get_rag_config()
    print("app.yml   :", app_cfg)
    print("chroma.yml:", rag_cfg)
    print("DB_PATH   :", app_cfg.abs_path("path.db"))
    print("TXT_PATH  :", app_cfg.abs_path("path.rag_text"))
    print("BASE_URL  :", app_cfg.get("llm.base_url"))
