

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

    def get_context(self, query: str)-> str:

        MAX_DISTANCE =float(RAG_CFG.get("max_distance",0.8))

        results = self.db.similarity_search_with_score(query, k= RAG_CFG.get("k", 3))

        valid_context=[]

        for i ,(doc,score) in enumerate(results):
            if float(score)>MAX_DISTANCE:
                logger.debug(f"⏭️ 丢弃低相关片段，距离超过阈值 {MAX_DISTANCE}")
                continue

            source_path=doc.metadata.get("source","未知来源")
            file_name=os.path.basename(source_path)
            logger.debug(f"候选片段 {i + 1} | 距离: {float(score):.4f} | 来源: {file_name}")

            chunk_text=f"[片段{i+1}] 来源 {file_name} \n{doc.page_content}"
            valid_context.append(chunk_text)

        if not valid_context:
            logger.warning("⚠️ 所有检索结果均被距离阈值过滤，未找到相关上下文。")
            return ""

        logger.info(f"✅ 检索完成，保留了 {len(valid_context)} 个有效片段。")
        return "\n\n---\n\n".join(valid_context)


if __name__ == '__main__':
    #实例化
    retriever=Retriever()

    query = "无线蓝牙耳机 Plus多少钱"
    print(f"🔍 搜索：{query}\n")

    final_context=retriever.get_context(query)

    print("================ 最终送给大模型的上下文 ================")
    print(final_context)
