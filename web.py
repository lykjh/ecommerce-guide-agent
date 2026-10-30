import streamlit as st

from agent.main_agent import get_ai_stream
#🔴 这里不能再用 nest_asyncio：Python 3.12+ 的 asyncio.timeout 要求在 Task 内运行，
#   而 nest_asyncio 会在执行 handle 前把当前 task 摘掉，langgraph 内部的
#   asyncio.wait_for 会直接抛 RuntimeError: Timeout should be used inside a task。
#   替代方案见 utils/async_bridge.py：后台线程常驻事件循环，主线程同步取 chunk。
from utils.async_bridge import iter_async

st.set_page_config(page_title="智能客服",page_icon="💬")
st.title("💬 智能客服小暖")
st.caption("🚀 基于 Qwen 大模型的实时对话")


#初始化聊天记录
if "message" not in st.session_state:
    st.session_state.message=[]

for msg in st.session_state.message:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt :=st.chat_input("说点什么吧..."):
    #显示用户的信息并保存
    st.session_state.message.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    #显示AI的信息并保存
    with st.chat_message("assistant"):
        response_placeholder=st.empty()
        full_response=""

        #iter_async 把 get_ai_stream 这个 async generator 搬到后台线程的事件循环里跑，
        #chunk 通过队列回到主线程，所以这里仍然是「边生成边刷新」的流式效果。
        for chunk in iter_async(get_ai_stream, prompt):
            full_response+=chunk
            #光标移动
            response_placeholder.markdown(full_response+"|")
        #结束时去掉光标
        response_placeholder.markdown(full_response)

    st.session_state.message.append({"role":"assistant","content":full_response})

