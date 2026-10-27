
from dotenv import load_dotenv
load_dotenv()

from utils.config_tool import get_app_config
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
import os
from agent.memory import AgentState, memory_manager
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rag.retriever import Retriever
retriever = Retriever()
from agent.router import router
from langgraph.graph import END


#配置统一从 config/app.yml 读取（原来写死在 mmm.py 里的常量）
APP_CFG = get_app_config()
#SQLite 记忆库绝对路径
DB_PATH = APP_CFG.abs_path("path.db")
#模型服务地址。
#⚠️ 地址不写在 yml 里（专属网关带租户标识，仓库公开会泄露），
#只从环境变量读：yml 里配的是变量名 llm.base_url_env，真实值放 .env
BASE_URL = os.getenv(APP_CFG.get("llm.base_url_env"))

client =init_chat_model(
    model=APP_CFG.get("llm.model", "qwen-plus"),
    model_provider=APP_CFG.get("llm.model_provider", "openai"),
    api_key=os.getenv(APP_CFG.get("llm.api_key_env", "OPENAI_API_KEY")),
    base_url=BASE_URL,
    temperature=APP_CFG.get("llm.temperature", 0.7),# 温度参数，控制生成文本的随机性
    max_tokens=APP_CFG.get("llm.max_tokens", 1024),# 最大生成 token 数，控制生成文本的长度
    #top_p=1,# 控制生成文本的多样性
    )

async def business_node(state: AgentState):
    user_query=state["messages"][-1].content
    context=retriever.get_context(user_query)

    msgs=await memory_manager.build_bussiness_messages(state,user_query,context)

    response = await client.ainvoke(msgs)
    return {"messages":[response]}


async def chat_node(state: AgentState):
    # 一行代码搞定闲聊历史
    msgs = await memory_manager.build_chat_messages(state)

    response = await client.ainvoke(msgs)
    return {"messages": [response]}


builder=StateGraph(AgentState)

#注册两个「干活」的节点：业务问答 / 闲聊
builder.add_node("business_node", business_node)
builder.add_node("chat_node", chat_node)
builder.add_node("summarize_node", memory_manager.summarize_node)

#⚠️ router 不能注册成普通节点：
#节点返回值必须是 dict（用来更新 state），而 router 返回的是 "闲聊"/"业务咨询" 这类字符串，
#注册成节点会直接报 InvalidUpdateError: Expected dict, got 闲聊。
#正确姿势是让它只当 START 的「分支函数」——只负责挑下一个节点，不产生 state 更新。
#这样每轮也只调用一次路由模型（注册成节点会被调用两次：进节点一次、判分支一次）。
builder.add_conditional_edges(
    START,#从入口直接分支，不经过多余的 router 节点
    router.router,
    {
        "闲聊": "chat_node",
        "业务咨询": "business_node",
        "其他": "chat_node"
    },
)

builder.add_edge("business_node", "summarize_node")
builder.add_edge("chat_node", "summarize_node")
builder.add_edge("summarize_node", END)
#本地。db文件存储作为记忆系统

async def get_ai_stream(user_input: str,thread_id:str="default"):
    config={"configurable":{"thread_id":thread_id}}

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph=builder.compile(checkpointer=checkpointer)

        async for msg_chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
        ):
            #⚠️ 光判 type 不够：流模式下所有节点的 LLM 输出都会被吐出来，
            #包括「路由模型」自己输出的 "闲聊"/"业务咨询"（它的归属节点是 __start__），
            #不按来源节点过滤的话，用户会看到回复开头挂着这两个意图标签。
            node_name=metadata.get("langgraph_node")
            if (
                msg_chunk.type == "AIMessageChunk"
                and msg_chunk.content
                and node_name in ("business_node", "chat_node")
            ):
                yield msg_chunk.content


if __name__ == "__main__":
    async def test():
        print("测试开始：")
        async for chunk in get_ai_stream("无线蓝牙耳机多少钱"):
            print(chunk, end="", flush=True)
        print("\n")

        # 测一下记忆：换个问法问上面那句话
        async for chunk in get_ai_stream("智能保温杯还有货吗"):
            print(chunk, end="", flush=True)
        print("\n")

    asyncio.run(test())


