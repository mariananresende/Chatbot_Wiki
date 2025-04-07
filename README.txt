# 📚 Chatbot Documenta Wiki

Este é um assistente conversacional baseado em LLM que responde dúvidas sobre a plataforma Documenta Wiki do MDS (Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome).

O projeto utiliza embeddings semânticos para recuperar trechos de manuais e documentos de referência da ferramenta, e gera respostas com base em um modelo de linguagem hospedado via API Groq.

## 🚀 Funcionalidades

- Responde perguntas sobre uso da Documenta Wiki
- Explica como editar, publicar e solicitar acesso
- Gera fichas de programas e indicadores com base em orientações (PDFs)
- Usa vetorização semântica para garantir respostas precisas
- Compatível com deploy no Streamlit Cloud

## 🧰 Tecnologias utilizadas

- [Langchain](https://python.langchain.com/)
- [Google Generative AI Embeddings (`embedding-001`)](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss) para vetores em memória
- [Groq API](https://console.groq.com/)
- [Streamlit](https://streamlit.io/) como interface

## ⚙️ Como rodar localmente

1. Clone o repositório:

```bash
git clone https://github.com/mariananresende/Chatbot_Wiki.git
cd Chatbot_Wiki

2. Crie um ambiente virtual e ative:

python -m venv venv
venv\\Scripts\\activate   # no Windows

3. Instale as dependências:

pip install -r requirements.txt

4. Crie um arquivo .env com sua chave da Groq e Google API:

groq_api_key=sk-xxxxxx
google_api_key=AIza...

5. Execute o app:

streamlit run app.py


☁️ Como publicar no Streamlit Cloud
Suba o repositório no GitHub

Acesse https://share.streamlit.io

Clique em “New app”

Escolha o repositório e o script app.py

Em Settings > Secrets, adicione:

groq_api_key = "sk-..."
google_api_key = "AIza..."


Clique em Deploy 🎉

📝 Observações
Ao solicitar a geração de uma ficha de indicador, o assistente usará o documento Ficha de Indicador.pdf como base e pedirá insumos necessários.

Para ficha de programa, é necessário fornecer referências legais e informações técnicas.

📄 Licença
MIT - Mariana N. Resende, 2025

Propostas de fichas geradas devem ser revisadas antes de uso oficial.


