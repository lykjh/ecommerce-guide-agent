import streamlit as st
import asyncio
import nest_asyncio

nest_asyncio.apply()
from agent.main_agent import get_ai_stream

st.set_page_config(page_title="智能客服",page_icon="💬")
st.title("💬 智能客服小暖")
var = st.caption
"🚀 基于 Qwen 大模型的实时对话"


#初始化聊天记录
if "message" not in st.session_state:
    st.session_state.message=[]

for msg in st.session_state.message:
    with st.chat_message(msg["role"]):
        st.markdown([msg["content"]])

if prompt :=st.chat_input("说点什么吧..."):
    #显示用户的信息并保存
    st.session_state.message.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    #显示AI的信息并保存
    with st.chat_message("assistant"):
        response_placeholder=st.empty()
        full_response=""

        async def fetch_ai():
            global full_response
            async for chunk in get_ai_stream(prompt):
                full_response+=chunk
                #光标移动
                response_placeholder.markdown(full_response+"|")
            #结束时去掉光标
            response_placeholder.markdown(full_response)
        #运行异步代码
        asyncio.run(fetch_ai())

    st.session_state.message.append({"role":"assistant","content":full_response})

