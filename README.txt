📚 Chatbot Documenta Wiki
Este é um assistente conversacional baseado em modelos de linguagem (LLM) voltado para auxiliar no uso da Documenta Wiki, plataforma oficial do Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS). Ele responde perguntas sobre a ferramenta, gera fichas de programas e indicadores, e orienta pontos focais quanto às regras de preenchimento.

🚀 Funcionalidades
🧠 Responde dúvidas sobre uso da Documenta Wiki

✏️ Explica como editar, publicar e solicitar acesso às fichas

📄 Gera propostas de fichas de programa e indicador com base em documentos oficiais (PDFs)

🧭 Utiliza vetorização semântica para garantir precisão nas respostas

☁️ Compatível com deploy no Streamlit Cloud

🧰 Tecnologias utilizadas
LangChain — para orquestração do RAG

Google Generative AI Embeddings (embedding-001) — para gerar vetores semânticos

FAISS — para indexação dos documentos

Groq API — hospedagem do modelo LLM (LLaMA 3)

Streamlit — como interface web

⚙️ Como rodar localmente
Clone o repositório:

bash
Copiar
Editar
git clone https://github.com/mariananresende/Chatbot_Wiki.git
cd Chatbot_Wiki
Crie um ambiente virtual e ative:

bash
Copiar
Editar
python -m venv venv
venv\Scripts\activate  # Windows
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
Crie um arquivo .env com suas chaves:

env
Copiar
Editar
groq_api_key=sk-xxxxxxx
google_api_key=AIza...
Execute o app:

bash
Copiar
Editar
streamlit run app.py
☁️ Como publicar no Streamlit Cloud
Suba o repositório para o GitHub

Acesse https://share.streamlit.io

Clique em “New app”

Escolha o repositório e o script app.py

Em Settings > Secrets, adicione:

text
Copiar
Editar
groq_api_key = "sk-..."
google_api_key = "AIza..."
Clique em Deploy 🎉

📝 Observações importantes
Ao solicitar a geração de uma ficha de indicador, o assistente usará o conteúdo do PDF Ficha de Indicador.pdf como referência e pedirá os insumos mínimos para preenchimento.

Para a ficha de programa, o usuário deverá fornecer referências legais e informações técnicas.

A proposta de nome de indicador segue as regras do Protocolo de Nomeação, integradas ao prompt.

Indicadores e fichas geradas são apenas sugestões e devem ser revisadas pela equipe técnica antes do uso oficial.

📄 Licença
MIT - Mariana N. Resende, 2025


