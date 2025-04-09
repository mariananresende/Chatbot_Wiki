import os
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"
import nltk

# Configuração do caminho local para os dados do NLTK
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

from dotenv import load_dotenv
import streamlit as st
import time

# === LlamaIndex ===
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    PromptTemplate,
)
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.google import GooglePaLMEmbedding

# === Carregar variáveis de ambiente ===
load_dotenv(dotenv_path="Chatbot_Wiki/.env")

groq_api_key = os.getenv("groq_api_key") or st.secrets.get("groq_api_key")
google_api_key = os.getenv("google_api_key") or st.secrets.get("google_api_key")

if not groq_api_key:
    st.error("❌ Chave da Groq não encontrada.")
    st.stop()

if not google_api_key:
    st.error("❌ Chave da Google API não encontrada.")
    st.stop()

# === Aparência da interface ===
st.set_page_config(page_title="Chat Documenta Wiki (LlamaIndex)", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        font-size: 16px !important;
        border-radius: 8px !important;
        padding: 10px;
    }
    .stButton > button {
        background-color: #1e467b;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.2rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #16395f;
    }
    .chat-box {
        background-color: #e9ecef;
        padding: 1rem;
        border-left: 5px solid #204d74;
        border-radius: 5px;
        margin-top: 1rem;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# === Cabeçalho ===
st.image("wiki.png", width=220)
st.title("Chat Documenta Wiki (LlamaIndex)")
st.caption("Tire dúvidas sobre a ferramenta de documentação oficial do MDS - versão LlamaIndex")

# === Modelos ===
embed_model = GooglePaLMEmbedding(model_name="models/embedding-001", api_key=your_api_key)
llm = Groq(model="llama3-8b-8192", api_key=groq_api_key)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# === Prompt personalizado ===
custom_prompt = PromptTemplate("""\
Você é um assistente especializado na Documenta Wiki, ferramenta oficial do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS), utilizada para documentar programas, ações, sistemas e indicadores.

Baseie sua resposta no contexto fornecido abaixo. Dê respostas completas, expandindo a explicação com base no conteúdo conhecido sobre a plataforma. Responda sempre em linguagem acessível, porém formal.

⚠️ Diferencie claramente:
- Quando a pergunta for sobre **como solicitar acesso para editar**, responda com o procedimento institucional (envio de e-mail ao DMA). Traga o prazo que o DMA tem para responder.
- Quando for sobre **como editar uma ficha**, apresente o passo a passo das instruções da interface.
- Quando for sobre **quem pode criar uma ficha de programa**, informe que para criar uma nova ficha de programa é preciso enviar solicitação ao DMA por e-mail. A ficha será criada após envio completo das informações.
- Quando for sobre **quem pode criar uma ficha de indicador**, informe que deve ser enviada solicitação ao DMA por e-mail. A ficha será criada após envio completo das informações em até 48 horas.
- Quando for sobre **quem pode publicar uma ficha**, diferencie claramente:
  - A **ficha de programa só pode ser publicada após análise e autorização prévia do DMA**, mesmo que tenha sido completamente preenchida pela área responsável.
  - A **ficha de indicador pode ser publicada diretamente pela área responsável**, **sem necessidade de autorização do DMA**, desde que esteja completamente preenchida conforme as orientações da plataforma. Essa autonomia visa dar mais dinamismo à documentação e reconhece o protagonismo técnico da área que gerencia o programa.

Se a pergunta solicitar **uma ficha de indicador preenchida**, use o documento base da ficha como referência. Avalie a orientação para preenchimento de cada campo contido no material e **solicite que o usuário forneça as informações mínimas necessárias para o preenchimento dos campos**. Tente, a partir do contexto dado, propor os campos de cada ficha. Para propor o nome do indicador, **utilize as regras do protocolo de nomeação**: tipo de medida + unidade + população-alvo + recorte geográfico ou temporal, se necessário. Destaque que o nome deve ser validado em conjunto com o DMA.

Se a pergunta envolver **como preencher um determinado campo da ficha do indicador**, descreva o que deve conter no campo questionado e sugira exemplos de resposta.

Se a pergunta envolver **propor uma ficha de programa preenchida**, destaque que é necessário o envio de **referências legais e informações técnicas** sobre o programa. Avalie a orientação para preenchimento de cada campo contido no material de referência.

🔎 Importante: Ao propor qualquer ficha preenchida, **informe que a proposta pode conter erros**, devendo ser revisada com atenção pelo ponto focal antes de ser transportada para a Documenta Wiki.

Se a pergunta for sobre conteúdos que mudam frequentemente (como lista de programas), oriente o usuário a acessar a Documenta Wiki pelo link oficial: mds.gov.br/documenta-wiki. Entretanto, explique a organização básica da ferramenta, com a apresentação dos programas atualmente vigentes e os programas descontinuados. Que ao acessar a página de cada programa é possível acessar a lista de indicadores documentados e outros conteúdos relacionados ao programa.

Nunca cite os nomes dos documentos utilizados como referência ao responder.

Sempre no final de cada interação, use frases motivacionais sobre a importância da documentação e da completude do preenchimento das fichas, variando a cada interação.

<context>
{context_str}
</context>

Pergunta:
{query_str}
""")

# === Documentos usados no RAG ===
pdf_files = [
    "Manual_de_Uso_Documenta_Wiki_MDS_SAGICAD.pdf",
    "Roteiro_video_divulgacao.pdf",
    "Roteiro_Tutorial_Documenta_Wiki.pdf",  # atualizado aqui
    "Ficha de Indicador.pdf",
    "Ficha de Programa.pdf",
    "Protocolo_nomeacao_indicadores.pdf"
]

if "index" not in st.session_state:
    if st.button("🔍 Carregar base de conhecimento"):
        with st.spinner("Processando documentos..."):
            try:
                docs = SimpleDirectoryReader(input_files=pdf_files).load_data()
                index = VectorStoreIndex.from_documents(docs, service_context=service_context)

                response_synthesizer = get_response_synthesizer(
                    service_context=service_context,
                    text_qa_template=custom_prompt
                )
                st.session_state.index = index
                st.session_state.synthesizer = response_synthesizer
                st.success("✅ Base carregada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar documentos: {e}")
                st.stop()

# === Entrada do usuário ===
question = st.text_area(
    "Digite sua pergunta sobre a Documenta Wiki:",
    placeholder="Ex: Quem pode publicar uma ficha de programa?",
    height=100
)

# === Geração da resposta ===
if question and "index" in st.session_state:
    with st.spinner("Gerando resposta..."):
        query_engine = st.session_state.index.as_query_engine(
            response_synthesizer=st.session_state.synthesizer
        )
        start = time.time()
        response = query_engine.query(question)
        elapsed = time.time() - start

    st.image("wiki.png", width=120)
    st.markdown(f"<div class='chat-box'>{response.response}</div>", unsafe_allow_html=True)
    st.caption(f"⏱️ Tempo de resposta: {elapsed:.2f} segundos")

    with st.expander("📄 Trechos usados da base"):
        for node in response.source_nodes:
            st.markdown(f"""
                <div style="background-color:#f0f0f0; padding:10px; margin:5px; border-left: 4px solid #888;">
                    <p>{node.node.text}</p>
                </div>
            """, unsafe_allow_html=True)
elif question:
    st.warning("⚠️ Clique em 'Carregar base de conhecimento' antes de perguntar.")

