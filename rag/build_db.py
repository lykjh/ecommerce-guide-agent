"""RAG 知识库构建：把 rag_text/（data_path）目录下的 txt 增量灌进向量库。

设计目标：做成**幂等的 sync**，而不是一次性 import。
    反复调用只处理「新增 / 内容有变更」的文件，未变动的直接跳过。
    数据源固定为 chroma.yml 的 data_path，网页端上传的文件也落到同一个目录，
    落盘后只要再调一次 build_db() 就会自动入库，不需要额外写增量逻辑。

整体流程：扫描 → 类型过滤 → md5 增量比对 → 切分 → 入库 → 回写 md5

日志：统一走 utils/log_tool.get_loger，同时输出到控制台和 logs/ 目录，
      不再用 print，避免「控制台有、文件没有」导致事后无法排查。
"""

import os
import json
import hashlib

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter

from utils.config_tool import get_app_config, get_rag_config
from utils.log_tool import get_loger

#应用配置：主要用来取 API Key 的环境变量名（读 config/app.yml）
APP_CFG = get_app_config()
#RAG 配置：向量库、切分参数、投放目录等（读 config/chroma.yml）
RAG_CFG = get_rag_config()

#本模块的日志记录器：控制台 + logs/rag.build_db_日期.log 双输出
LOGGER = get_loger("rag.build_db")


class VectorStoreService:
    def __init__(self):
        # 初始化embedding模型（百炼 DashScope 兼容 OpenAI 协议）
        # API Key 从环境变量取，不写死在配置文件里，避免密钥进仓库
        self.embeddings = DashScopeEmbeddings(
            model=RAG_CFG.get("embedding_model", "text-embedding-v4"),
            dashscope_api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY"))
        )

        #从 config/chroma.yml 读取参数，yml 里的相对路径统一转成项目根下的绝对路径
        #集合名：同一份知识库共用一个集合，改名字等于换库
        self.collection_name=RAG_CFG.get("collection_name", "agent")
        #向量库落地目录
        self.persist_directory=RAG_CFG.abs_path("persist_directory")
        #知识文档投放目录：网页端上传的文件往这里塞，就会被自动扫到并入库
        self.data_path=RAG_CFG.abs_path("data_path")
        #md5 记录文件：保存「文件绝对路径 -> 文件内容 md5」，用来判断哪些文件变过
        self.hash_file_path=RAG_CFG.abs_path("md5_hex_store")

        #文本切分器
        #注意：chunk_size / chunk_overlap 必须是 int，早期版本用 os.getenv() 取出来是字符串会报错
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=RAG_CFG.get("CHUNK_SIZE", 200),
            chunk_overlap=RAG_CFG.get("CHUNK_OVERLAP", 40),
            separators=RAG_CFG.get("SEPARATORS", ["\n\n", "\n", "。", "！", "？", "，", " "])
        )

        LOGGER.debug(
            f"VectorStoreService 初始化完成 | 集合={self.collection_name} | "
            f"向量库={self.persist_directory} | 数据源={self.data_path}"
        )

    def get_file_md5(self,file_path:str)->str:
        """
        计算单个文件内容的 md5（按内容算：改名、改修改时间都不算变更，只有内容变了值才变）
        :param file_path: 文件绝对路径
        :return: md5 十六进制字符串
        """
        md5=hashlib.md5()
        #二进制分块读取，避免大文件一次性载入内存（products.txt 有 268KB，更大的也不怕）
        with open(file_path,"rb") as f:
            while True:
                chunk=f.read(8192)
                #读不到内容说明到文件末尾，break 退出
                #坑：别写成 iter(lambda: f.read(8192), "b")，哨兵必须是「空字节」 b""，
                #写成字符串 "b" 永远不相等，会导致死循环
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()

    def get_all_files(self,path: str) -> list[str]:
        """
        把传入路径统一展开成文件列表：是文件就返回自己，是目录就递归收集全部文件
        :param path: 文件或目录路径
        :return: 文件路径列表（已做 normpath 规范化）
        """
        file_list=[]
        if os.path.isfile(path):
            #单个文件：包成只含一个元素的列表，后面的处理逻辑就能复用同一套
            file_list.append(path)
        elif os.path.isdir(path):
            #os.walk 递归遍历所有子目录
            #sorted 排序保证多次扫描顺序稳定，日志和 md5 记录不会因为顺序变化而抖动
            for root,_,files in os.walk(path):
                for file in sorted(files):
                    file_list.append(os.path.join(root,file))
        else:
            #路径不存在直接报错，不要静默返回空列表（那会让人误以为是"目录里没文件"）
            raise FileNotFoundError(f"❌ 路径既不是文件也不是目录: {path}")
        #统一路径分隔符：Windows 下 os.walk 出来是反斜杠，而手写在配置/脚本里的常是正斜杠，
        #不规范化会被当成两个不同的 key，导致 md5 比对永远匹配不上、去重失效
        return [os.path.normpath(p) for p in file_list]

    def _allowed_types(self)->tuple:
        """
        把配置里的后缀列表规整成可直接给 str.endswith 用的元组
        :return: 例如 ('.txt',)
        """
        raw_types=RAG_CFG.get("allow_knowledge_file_type", ["txt"])
        #去掉可能带的前导点并统一转小写，配置里写 "txt" / ".txt" / "TXT" 都能识别
        return tuple(f".{ext.lower().lstrip('.')}" for ext in raw_types)

    def _load_md5_record(self)->dict:
        """
        读取 md5 记录文件
        :return: {文件绝对路径: md5}，文件不存在或内容损坏时返回空 dict（等于全量重处理）
        """
        if not os.path.exists(self.hash_file_path):
            LOGGER.info("未发现 md5 记录文件，本次将全量处理。")
            return {}
        try:
            with open(self.hash_file_path,"r",encoding="utf-8") as f:
                record=json.load(f)
            #历史记录的 key 也要规范化：可能是旧版本写的、或人手填的正斜杠路径，
            #不统一就永远和扫描出来的路径对不上，去重会静默失效
            record={os.path.normpath(k):v for k,v in record.items()}
            LOGGER.debug(f"已加载 md5 记录，共 {len(record)} 条。")
            return record
        except (json.JSONDecodeError,OSError) as e:
            #记录文件坏了不要崩，降级成全量重处理（代价只是多花一次 embedding 费用）
            LOGGER.warning(f"⚠️ md5 记录文件损坏，将重新全量处理: {e}")
            return {}

    def _save_md5_record(self,record: dict)->None:
        """
        回写 md5 记录
        关键：必须在「入库成功之后」才调用。如果入库前就写，中途失败会导致下次误判成已处理，
        那个文件就永久进不了库了
        :param record: {文件绝对路径: md5}
        """
        #父目录兜底创建，避免 md5 记录放在不存在的子目录时报错
        os.makedirs(os.path.dirname(self.hash_file_path),exist_ok=True)
        with open(self.hash_file_path,"w",encoding="utf-8") as f:
            json.dump(record,f,ensure_ascii=False,indent=2)

    def build_db(self):
        """
        构建 / 同步知识库（幂等：反复调用只处理新增和变更的文件）
        数据源固定用 chroma.yml 的 data_path（rag_text/）：
        网页端把上传的文件存进这个目录，再调一次本方法即可自动入库
        :return: 本次实际入库的文件数
        """
        LOGGER.info(f"🚀 开始检查并构建知识库，数据源: {self.data_path}")

        all_files=self.get_all_files(self.data_path)
        if not all_files:
            raise ValueError(f"❌ 没有找到任何文件，请检查目录下是否有文件: {self.data_path}")
        LOGGER.info(f"🚀 共找到 {len(all_files)} 个文件。")

        #允许入库的文件后缀，例如 ('.txt',)
        allow_types=self._allowed_types()

        #Chroma 实例在循环外建一次：批量入库时不能每个文件都重新初始化一遍
        vector_store =Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        #上次处理记录：{文件绝对路径: md5}
        processed_files=self._load_md5_record()
        done_count=0
        skip_count=0

        for file_path in all_files:
            #类型过滤：只处理配置里允许的后缀（比较前先 lower，避免 .TXT 被漏掉）
            if not file_path.lower().endswith(allow_types):
                LOGGER.debug(f"跳过非目标类型文件: {file_path}")
                continue

            #按「文件内容」算 md5，内容没变值就不变
            current_hash=self.get_file_md5(file_path)

            #和上次记录一致 → 未变更，跳过
            #这一步是省钱的：embedding 要调 API，重复处理既费时间又费钱
            if processed_files.get(file_path)==current_hash:
                LOGGER.info(f"✅ 文件未修改，跳过: {os.path.basename(file_path)}")
                skip_count+=1
                continue

            LOGGER.info(f"🚀 处理文件: {os.path.basename(file_path)}")

            try:
                loader =TextLoader(file_path,encoding="utf-8")
                docs=loader.load()

                #给每个文档打上来源（用规范化后的路径，和 md5 记录的 key 保持一致），
                #切分后每个文本块都会继承这份元数据
                #键名必须是 source：写成 score 会和检索时的「相似度分数」撞车，
                #而且元数据是建库时写入的，事后想补只能重建整个库
                for doc in docs:
                    doc.metadata["source"]=file_path

                splits=self.splitter.split_documents(docs)

                #给每个文本块指定稳定 id：文件md5_序号
                #这样万一 md5 记录丢了要全量重灌，相同 id 会覆盖而不是追加，库不会膨胀出重复数据
                ids=[f"{current_hash}_{i}" for i in range(len(splits))]

                vector_store.add_documents(splits,ids=ids)
                LOGGER.info(f"✅ 处理完成: {os.path.basename(file_path)}，新增 {len(splits)} 个文本块")

                #入库成功后才回写 md5
                processed_files[file_path]=current_hash
                self._save_md5_record(processed_files)
                done_count+=1

            except UnicodeDecodeError:
                #txt 不是 UTF-8（常见是 GBK 编码），TextLoader 读不了
                #不中断整批，只记录，用户转成 UTF-8 后下次跑会自动补上
                LOGGER.error(f"❌ 文件不是 UTF-8 编码，无法读取: {file_path}")
            except Exception as e:
                #单个文件失败不影响整批，记录后继续处理下一个
                #用 exception 会带上完整堆栈，方便定位
                LOGGER.exception(f"❌ 处理文件 {file_path} 时出错: {e}")

        LOGGER.info(
            f"✅ 知识库同步完成！本次入库 {done_count} 个文件，跳过 {skip_count} 个未变更文件。"
        )
        return done_count


if __name__ == "__main__":
    # 固定同步 chroma.yml 里 data_path 指定的目录（rag_text/）
    # 注意必须用 -m 方式运行：模块里有 from utils.xxx 的包内导入，
    # 直接 python rag/build_db.py 会 ModuleNotFoundError（和 log_tool 是同一个坑）
    service = VectorStoreService()
    service.build_db()
