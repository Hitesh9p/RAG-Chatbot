# 📄 RAG Chatbot with LangChain, Groq, and Streamlit

This project is an interactive, memory-enabled RAG (Retrieval-Augmented Generation) chatbot built with the latest **LangChain composable architecture**, **Groq LLMs**, and **Streamlit**. Users can upload a PDF file and ask questions based on the document content — with support for follow-up queries using conversational memory.

---

## 🚀 Features

- 🔍 Question-answering on uploaded PDFs
- 🧠 Multi-turn conversation with memory (`ConversationBufferMemory`)
- ⚡ Super-fast LLM responses using Groq's `llama3-70b-8192`
- 🧩 Modular, future-proof LangChain pipeline with latest packages
- 📦 Local embedding support with `Ollama` (`nomic-embed-text`)
- 🗂 Vector search powered by `Chroma`

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/) (latest composable architecture)
- [Streamlit](https://streamlit.io/) (interactive UI)
- [Groq API](https://console.groq.com/) for LLM responses
- [Ollama](https://ollama.com/) for embeddings
- [Chroma](https://www.trychroma.com/) as vector store

---

## ⚙️ Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot-groq
cd rag-chatbot-groq

2. Install Dependencies
pip install streamlit langchain langchain-community langchain-core langchain-text-splitters langchain-groq python-dotenv chromadb

3. Set Up .env File
Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

4. Run the App
streamlit run app.py
