# 🤖 Chatbot Documenta Wiki - MDS

Este é um aplicativo Streamlit que utiliza **RAG (Retrieval-Augmented Generation)** com a LLM LLaMA3 via Groq para responder dúvidas sobre a **Documenta Wiki**, ferramenta do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS).

## 📚 Fontes de conhecimento utilizadas

O sistema utiliza os seguintes documentos oficiais como base de conhecimento:

- Manual de Uso da Documenta Wiki - MDS_SAGICAD.pdf
- Manual de Uso da Documenta Wiki - Teste.pdf
- Roteiro video divulgacao.pdf
- Roteiro_Tutorial_Documenta_Wiki.pdf (roteiro gerado a partir do vídeo institucional)

## 🚀 Tecnologias utilizadas

- Streamlit
- LangChain
- ChromaDB (vector store)
- Hugging Face Embeddings (MiniLM-L6-v2)
- LLM da Groq (LLaMA3-8B)
- PyPDFLoader

## 🧠 Como funciona

1. O usuário carrega os documentos com o botão “Carregar base do chat”
2. O sistema realiza a vetorização (embeddings + indexação)
3. O usuário faz perguntas via caixa de entrada
4. A LLM responde com base nos trechos dos documentos
5. Quando a pergunta exige informações externas ou atualizadas (como lista de programas), a IA orienta a acessar o portal da Documenta Wiki

## 🛠️ Instalação

1. Clone o repositório
2. Crie um ambiente virtual e ative-o
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Adicione sua chave da API da Groq em um arquivo `.env`:

```
groq_api_key=SUA_CHAVE_AQUI
```

5. Rode o aplicativo:

```bash
streamlit run app.py
```

## 🌐 Acesso oficial à plataforma

[https://wiki-sagi.cidadania.gov.br](https://wiki-sagi.cidadania.gov.br)

## 📩 Contato institucional

Em caso de dúvidas, erros ou sugestões, contate:  
**wiki@mds.gov.br**
