import streamlit as st
import os
import time

# Langchain e integração
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

# Interface
st.image("wiki.png", width=200)
st.title("Chat Documenta Wiki - Dúvidas sobre a ferramenta")

# LLM via Groq
groq_api_key = os.getenv('groq_api_key')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt orientado
prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado na Documenta Wiki, ferramenta oficial do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS), utilizada para documentar programas, ações, sistemas e indicadores.

Baseie sua resposta no contexto fornecido abaixo. De respostas completas, expandindo a explicação com base no conteúdo conhecido sobre a plataforma e de maneira institucional.

⚠️ Diferencie claramente:
- Quando a pergunta for sobre **como solicitar acesso para editar** (perfil de edição), responda com o procedimento institucional (envio de e-mail ao DMA).
- Quando for sobre **como editar uma ficha**, use a orientação contida no material dado. 
- Quando for sobre **quem pode publicar uma ficha de programa**, destaque que após o preenchimento completo da ficha pelo ponto focal formalmente instituído a publicação depende da análise e autorização do DMA.
- Quando for sobre **quem pode publicar uma ficha de indicador**, destaque que a própria área deve publicar, desde que a ficha esteja completa e preenchida conforme as orientações constantes na ficha original. Destaque a importância da área preencher de maneira completa
as fichas dos indicadores e de no final publicar a ficha, para que quando o usuário for acessá-la não apareça a tela de erro (que significa que a ficha não está pública).

Se a pergunta for sobre conteúdos que mudam frequentemente (como lista de programas disponíveis), oriente o usuário a acessar diretamente a Documenta Wiki pelo link oficial:

mds.gov.br/documenta-wiki

Neste caso, explique como a página está organizada, destacando que na primeira página o usuário deve acessar a lista de programas vigente, e que dentro da página de cada programa, é possível acessar a lista de indicadores documentados, as fichas de sintaxe dos indciadores, 
dentre outros conteúdos relevantes sobre o programa.

Sempre destaque no final de cada resposta a importância da documentação de todos os indicadores disponibilizados, e do papel fundamental do ponto focal para garantir a qualidade do metadado de cada indicador e a sua disponibilidade para todos os interessados.

<contexto>
{context}
</contexto>

Pergunta:
{input}
""")

# Vetorização
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        docs = []

        # PDFs base
        pdf_paths = [
            "./Manual de Uso da Documenta Wiki _ MDS_SAGICAD.pdf",
            "./Manual de Uso da Documenta Wiki - Teste _ MDS_SAGICAD.pdf",
            "./Roteiro video divulgacao.pdf",
            "./Roteiro_Tutorial_Documenta_Wiki.pdf"
        ]
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        # 🔹 Documento manual: descrição geral
        descricao_documenta = Document(
            page_content="""
A Documenta Wiki é uma biblioteca online com informações sobre os programas e indicadores do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS), construída de forma colaborativa pelos pontos focais e gerenciada pelo Departamento de Monitoramento e Avaliação (DMA) da Secretaria de Avaliação, Gestão da Informação e Cadastro Único (SAGICAD).

Ela visa promover a transparência e otimizar os processos internos do Ministério, por meio da disponibilização de informações atualizadas sobre programas, ações, sistemas, projetos, atividades, indicadores de monitoramento e bases de dados. A plataforma é desenvolvida com a ferramenta Wiki.js.
""",
            metadata={"source": "inserido_manual"}
        )

        # 🔹 Documento manual: perfil de editor
        perfil_editor = Document(
            page_content="""
Solicitação de perfil de editor na Documenta Wiki:

Se for seu primeiro acesso com perfil de leitura, solicite a alteração para perfil de editor pelo e-mail:
wiki@mds.gov.br

Informe no e-mail:
- Nome completo
- E-mail institucional
- Área
- Login
- Ramal

O Departamento de Monitoramento e Avaliação (DMA) informará sobre a alteração em até 24 horas.

⚠️ O ponto focal com perfil de edição deve estar formalmente indicado.

O acesso à plataforma para edição se dá por:
https://mds.gov.br/documenta-wiki

Na tela de login, selecione o provedor “LDAP Active Directory” e use seu usuário e senha da rede do MDS.
""",
            metadata={"source": "inserido_manual"}
        )

        # 🔹 Documento manual: publicação de programa
        publicacao_programa = Document(
            page_content="""
Publicação de ficha de programa na Documenta Wiki:

A criação de ficha de programa deve ser solicitada por e-mail ao DMA (Departamento de Monitoramento e Avaliação). Após o envio completo das informações pela área responsável, o DMA criará a ficha.

A publicação da ficha só poderá ocorrer após:

1. Preenchimento completo pela área responsável
2. Comunicação ao DMA
3. Revisão dos dados pela equipe do DMA
4. Autorização formal do DMA

Somente após essa etapa a ficha será publicada na plataforma.
""",
            metadata={"source": "inserido_manual"}
        )

        # 🔹 Documento manual: publicação de indicador
        publicacao_indicador = Document(
            page_content="""
Publicação de ficha de indicador na Documenta Wiki:

Indicadores já cadastrados na plataforma podem ser publicados diretamente pela área responsável, desde que todos os campos da ficha estejam preenchidos corretamente.

Fluxo padrão:
1. Acesse a ficha do indicador no sistema
2. Verifique se todos os campos obrigatórios estão preenchidos
3. Caso estejam completos, a própria área pode proceder com a publicação imediata

⚠️ Caso o indicador ainda não tenha ficha criada, é necessário solicitar a criação ao DMA, que disponibilizará a estrutura da ficha. Após isso, a área tem até 10 dias úteis para preenchimento e publicação.
""",
            metadata={"source": "inserido_manual"}
        )

        docs.extend([
            descricao_documenta,
            perfil_editor,
            publicacao_programa,
            publicacao_indicador
        ])

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        st.session_state.vectors = Chroma.from_documents(
            chunks,
            st.session_state.embeddings,
            collection_name="documenta",
            persist_directory=None
        )
        st.session_state.ready = True

# Entrada
prompt1 = st.text_input("Digite sua pergunta sobre a Documenta Wiki")

# Botão
if st.button("Carregar base do chat"):
    vector_embedding()
    if st.session_state.get("ready"):
        st.success("✅ Chat carregado com sucesso e pronto para perguntas!")

# Execução
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























            




            
            

    
             
    

    
    



    
    
    
    
    



    
    
        
        

