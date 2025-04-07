from dotenv import load_dotenv
import streamlit as st
import os
import time

# Langchain e integração
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# === Carregar chaves ===
load_dotenv(dotenv_path="Chatbot_Wiki/.env")

groq_api_key = os.getenv("groq_api_key") or st.secrets.get("groq_api_key")
google_api_key = os.getenv("google_api_key") or st.secrets.get("google_api_key")

if not groq_api_key:
    st.error("⚠️ Chave da Groq não encontrada.")
    st.stop()

if not google_api_key:
    st.error("⚠️ Chave da Google API não encontrada.")
    st.stop()

# === Interface ===
st.image("wiki.png", width=200)
st.title("Chat Documenta Wiki - Dúvidas sobre a ferramenta")

# === LLM ===
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# === Prompt ===
prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado na Documenta Wiki, ferramenta oficial do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS), utilizada para documentar programas, ações, sistemas e indicadores.

Baseie sua resposta no contexto fornecido abaixo...

<contexto>
{context}
</contexto>

Pergunta:
{input}
""")

# === Função de vetorização ===
def vector_embedding():
    if "vectors" in st.session_state:
        return  # Já carregado

    try:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
    except Exception as e:
        st.error(f"❌ Erro ao inicializar embeddings: {e}")
        st.stop()

    pdf_paths = [
        "Manual_de_Uso_Documenta_Wiki_MDS_SAGICAD.pdf",
        "Manual_de_Uso_Documenta_Wiki_Teste_MDS_SAGICAD.pdf",
        "Roteiro_video_divulgacao.pdf",
        "Roteiro_video_tutorial_edicao.pdf",
        "Ficha de Indicador.pdf",
        "Ficha de Programa.pdf",
        "Ficha de Sintaxe.pdf"
    ]

    docs = []
    for path in pdf_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            st.warning(f"⚠️ Arquivo não encontrado: {path}")

    if not docs:
        st.error("❌ Nenhum documento foi carregado.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    def limpar_texto(txt):
        return txt.encode("utf-8", "ignore").decode("utf-8").strip()

    chunks = [
        Document(page_content=limpar_texto(doc.page_content), metadata=doc.metadata)
        for doc in chunks
    ]

    st.write(f"📄 {len(docs)} documentos carregados | 🔢 {len(chunks)} chunks gerados")

    try:
        _ = st.session_state.embeddings.embed_documents(["teste simples"])
    except Exception as e:
        st.error(f"❌ Falha ao testar embedding: {e}")
        st.stop()

    try:
        st.session_state.vectors = FAISS.from_documents(chunks, st.session_state.embeddings)
        st.session_state.ready = True
        st.success("✅ Vetorização concluída com sucesso!")
    except Exception as e:
        st.error(f"❌ Falha ao criar índice FAISS: {e}")
        st.stop()

# === Entrada do usuário ===
prompt1 = st.text_input("Digite sua pergunta sobre a Documenta Wiki")

# === Botão de carregamento ===
if st.button("Carregar base do chat"):
    with st.spinner("Carregando base de conhecimento..."):
        vector_embedding()

# === Execução do chat ===
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Clique em 'Carregar base do chat' antes de perguntar.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("🧠 Gerando resposta..."):
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt1})
            elapsed = time.process_time() - start

        st.markdown("### 🤖 Resposta")
        st.success(response['answer'])
        st.write(f"⏱️ Tempo de resposta: {elapsed:.2f} segundos")

        with st.expander("📄 Trechos usados da base de conhecimento"):
            for i, doc in enumerate(response.get("context", [])):
                st.markdown(f"""
                    <div style="background-color:#f0f0f0; padding:10px; margin:5px; border-left: 4px solid #888;">
                        <p>{doc.page_content}</p>
                    </div>
                """, unsafe_allow_html=True)

        

