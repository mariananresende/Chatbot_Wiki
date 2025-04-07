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

# === Carregar chave da API ===
load_dotenv(dotenv_path="Chatbot_Wiki/.env")

# Chave da LLM
groq_api_key = os.getenv("groq_api_key") or st.secrets.get("groq_api_key")
if not groq_api_key:
    st.error("⚠️ A chave da API da Groq não foi encontrada. Verifique o .env ou o secrets.toml.")
    st.stop()

# Chave da Google API
google_api_key = os.getenv("google_api_key") or st.secrets.get("google_api_key")
if not google_api_key:
    st.error("⚠️ A chave da API da Google (para embeddings) não foi encontrada. Verifique o .env ou o secrets.toml.")
    st.stop()

# === Interface ===
st.image("wiki.png", width=200)
st.title("Chat Documenta Wiki - Dúvidas sobre a ferramenta")

# === LLM ===
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# === Prompt ===
prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado na Documenta Wiki, ferramenta oficial do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS), utilizada para documentar programas, ações, sistemas e indicadores.

Baseie sua resposta no contexto fornecido abaixo. Se necessário para dar uma resposta mais completa, você pode expandir a explicação com base no conteúdo conhecido sobre a plataforma.

⚠️ Diferencie claramente:
- Quando a pergunta for sobre **como solicitar acesso para editar** (perfil de edição), responda com o procedimento institucional (envio de e-mail ao DMA).
- Quando for sobre **como editar uma ficha**, apresente as instruções da interface.
- Quando for sobre **quem pode publicar uma ficha de programa**, destaque que a publicação depende de autorização do DMA.
- Quando for sobre **quem pode publicar uma ficha de indicador**, destaque que a própria área pode publicar, desde que a ficha esteja completa.

Se a pergunta envolver **propor uma ficha de indicador preenchida**, use o documento "Ficha de Indicador.pdf" como base e **solicite que o usuário forneça as informações necessárias para o preenchimento dos campos**.

Se a pergunta envolver **propor uma ficha de programa preenchida**, destaque que é necessário o envio de **referências legais e informações técnicas** sobre o programa.

🔎 Ao propor qualquer ficha preenchida, **use como referência os documentos de orientação fornecidos** e **informe que a proposta pode conter erros**, devendo ser revisada com atenção pelo ponto focal antes de ser transportada para a Documenta Wiki.

Se a pergunta for sobre conteúdos que mudam frequentemente (como lista de programas), oriente o usuário a acessar a Documenta Wiki pelo link oficial:
https://wiki-sagi.cidadania.gov.br

<contexto>
{context}
</contexto>

Pergunta:
{input}
""")

# === Função de vetorizacão ===
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="embedding-001",
            google_api_key=google_api_key
        )

        docs = []
        pdf_paths = [
            "Manual_de_Uso_Documenta_Wiki_MDS_SAGICAD.pdf",
            "Manual_de_Uso_Documenta_Wiki_Teste_MDS_SAGICAD.pdf",
            "Roteiro_video_divulgacao.pdf",
            "Roteiro_video_tutorial_edicao.pdf",
            "Ficha de Indicador.pdf",
            "Ficha de Programa.pdf",
            "Ficha de Sintaxe.pdf"
        ]

        for path in pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            else:
                st.warning(f"Arquivo não encontrado: {path}")

        st.write(f"📄 Total de páginas extraídas: {len(docs)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)

        # Limpeza dos textos
        def limpar_texto(texto):
            return texto.encode("utf-8", "ignore").decode("utf-8").strip()

        chunks = [
            Document(page_content=limpar_texto(doc.page_content), metadata=doc.metadata)
            for doc in chunks
        ]

        st.write(f"🔢 Total de chunks: {len(chunks)}")

        # Teste de embedding com captura de erro
        try:
            _ = st.session_state.embeddings.embed_documents(["teste simples"])
        except Exception as e:
            st.error(f"❌ Erro na chamada de teste ao embedding: {e}")
            st.stop()

        try:
            st.session_state.vectors = FAISS.from_documents(
                chunks,
                st.session_state.embeddings
            )
            st.session_state.ready = True
        except Exception as e:
            st.error(f"❌ Erro ao criar vetor FAISS: {e}")
            st.stop()


# === Entrada ===
prompt1 = st.text_input("Digite sua pergunta sobre a Documenta Wiki")

# === Botão ===
if st.button("Carregar base do chat"):
    with st.spinner("Carregando vetores da base de conhecimento..."):
        vector_embedding()
    if st.session_state.get("ready"):
        st.success("✅ Chat carregado com sucesso e pronto para perguntas!")

# === Execução ===
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Por favor, clique em 'Carregar base do chat' antes de perguntar.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

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























            




            
            

    
             
    

    
    



    
    
    
    
    



    
    
        
        

