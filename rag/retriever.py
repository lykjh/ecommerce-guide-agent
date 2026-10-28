

from dotenv import load_dotenv
load_dotenv()
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_tool import get_app_config,get_rag_config
from utils.log_tool import get_loger
logger = get_loger("rag.retriever")


APP_CFG = get_app_config()
RAG_CFG = get_rag_config()
class Retriever:

    def __init__(self):

         #初始化Embedding
        self.embeddings = DashScopeEmbeddings(
            model =RAG_CFG.get("embedding_model","text-embedding-v4"),
            dashscope_api_key=os.getenv(APP_CFG.get("llm.api_key_env"))
        )

         #连接数据库
        self.db=Chroma(
            collection_name=RAG_CFG.get("collection_name","agent"),
            persist_directory=RAG_CFG.abs_path("persist_directory"),
            embedding_function=self.embeddings
        )
        try:
            cont =self.db._collection.count()
            if cont ==0:
                logger.warning(f"⚠️ 向量库为空！请检查路径是否正确：{RAG_CFG.abs_path('persist_directory')}")
            else:
                logger.info(f"向量库已存在，包含 {cont} 个向量")
        except Exception as e:
            #这里必须用模块级 logger，不能写 get_loger()：不传 name 会走默认名 'agent'，
            #日志会落到 logs/agent_日期.log，且 logger 名认不出来是自个模块的
            logger.error(f"向量库连接异常：{e}")

    def _metric_space(self) -> str:
        """向量库实际使用的距离度量。

        Chroma 创建集合时若不传 metadata，hnsw:space 默认就是 l2（平方欧氏距离），
        那个值可以远大于 1，跟「余弦距离」不是同一个量纲，混用会让阈值彻底失真。
        """
        try:
            meta = self.db._collection.metadata or {}
        except Exception as e:
            logger.warning(f"⚠️ 读取集合 metadata 失败，按 l2 处理: {e}")
            return "l2"
        return str(meta.get("hnsw:space") or "l2").lower()

    def _to_cosine_distance(self, raw_score) -> float:
        """把 similarity_search_with_score 的原始分数统一折算成「余弦距离」。

        余弦距离 = 1 - 余弦相似度，范围 0~2，越小越像。这样上层的阈值配置才有稳定语义，
        换 embedding 模型或重建库（换度量）都不用重新标定阈值。

        - 空间就是 cosine：原样返回
        - 空间是 l2 且向量已归一化（本库 embedding 模长实测 = 1.0）：
              d² = 2 - 2·cos  →  余弦距离 = 1 - cos = d² / 2
        """
        raw = float(raw_score)
        if self._metric_space() == "cosine":
            return raw
        return raw / 2.0

    def get_context(self, query: str, max_distance: float | int = None, k: int = None) -> str:

        # 2. 核心逻辑：如果你传了值，我就用你的；如果没传(None)，我才去读配置
        if max_distance is None:
            max_distance = float(RAG_CFG.get("max_distance", 0.5))
        if k is None:
            k = RAG_CFG.get("k", 3)

        results = self.db.similarity_search_with_score(query, k=k)

        logger.info(
            f"🔍 检索 | 查询「{query[:30]}」| 度量 {self._metric_space()} | "
            f"阈值(余弦距离) {max_distance} | k={k}"
        )

        valid_context=[]

        for i ,(doc,score) in enumerate(results):
            #统一折成余弦距离再比阈值，避免 l2 / cosine 量纲混用
            cos_dist=self._to_cosine_distance(score)

            if cos_dist > max_distance:
                logger.debug(f"⏭️ 丢弃片段 {i+1}：余弦距离 {cos_dist:.4f} > 阈值 {max_distance}")
                continue

            source_path=doc.metadata.get("source","未知来源")
            file_name=os.path.basename(source_path)
            logger.debug(
                f"✅ 保留片段 {i+1} | 余弦距离 {cos_dist:.4f} | 原始分 {float(score):.4f} | 来源 {file_name}"
            )

            chunk_text=f"[片段{i+1}] 来源 {file_name} \n{doc.page_content}"
            valid_context.append(chunk_text)

        if not valid_context:
            #把真实分数打出来，否则调阈值只能靠猜
            raw=[round(self._to_cosine_distance(s),4) for _,s in results]
            logger.warning(
                f"⚠️ 检索无有效结果 | 查询「{query[:30]}」| 阈值 {max_distance} | "
                f"各片段余弦距离 {raw}（全部超阈值，可考虑放宽 max_distance）"
            )
            return ""

        logger.info(f"✅ 检索完成，保留 {len(valid_context)}/{len(results)} 个片段。")
        return "\n\n---\n\n".join(valid_context)


#懒加载单例：向量库连接只要建一次。
#放在这里而不是各个使用方模块里，是为了避免 tool/ 和 agent/ 各自 new 一个 Retriever，
#变成两条重复的向量库连接。
_retriever_instance = None


def get_retriever() -> Retriever:
    """获取全局唯一的 Retriever 实例（首次调用时才真正连库）"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


if __name__ == '__main__':
    #实例化
    retriever=Retriever()

    query = "无线蓝牙耳机 Plus多少钱"
    print(f"🔍 搜索：{query}\n")

    final_context=retriever.get_context(query)

    print("================ 最终送给大模型的上下文 ================")
    print(final_context)
