import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st

st.set_page_config(page_title="Chatbot movido", page_icon="🔁")

st.image("wiki.png", width=220)
st.title("Chat Documenta Wiki")
st.caption("Tire dúvidas sobre como documentar os programas e respectivos indicadores na ferramenta de metadados oficial do MDS")

# === Cabeçalho ===

st.set_page_config(page_title="Chatbot movido", page_icon="🔁")

st.markdown("""
# 🔁 Chatbot movido!

Este chatbot foi transferido para um novo endereço.

👉 [Clique aqui para acessar o novo app](https://chatbotwiki.streamlit.app/)

Se você chegou aqui por um link antigo, atualize seus favoritos!
""")
