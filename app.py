import streamlit as st

st.title("你好，Streamlit Cloud！")
st.write("这是我第一个 Streamlit 应用。")

name = st.text_input("请输入你的名字：")
if name:
    st.success(f"你好，{name}！欢迎使用 Streamlit 🎈")
