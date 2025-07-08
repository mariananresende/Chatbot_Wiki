import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st

# Esta chamada deve aparecer apenas uma vez, logo após os imports
st.set_page_config(page_title="Chatbot movido", page_icon="🔁")

# Cabeçalho visual (opcional)
st.image("wiki.png", width=220)
st.title("Chat Documenta Wiki")

# Mensagem de redirecionamento
st.markdown("""
# 🔁 Chatbot movido!

Este chatbot foi transferido para um novo endereço.

👉 [Clique aqui para acessar o novo app](https://chatbotwiki.streamlit.app/)

Se você chegou aqui por um link antigo, atualize seus favoritos!
""")

