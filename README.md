# 🧠 Simple RAG Chatbot with Google Generative AI

A lightweight Retrieval-Augmented Generation (RAG) chatbot that uses Google Generative AI to answer questions from your own documents.  
Built with **LangChain**, **PostgreSQL (pgvector)**, and **Google Gemini models**.

---

## 🚀 Features
- **Document ingestion** — load and preprocess your PDFs or text files.
- **Vector storage** — store embeddings in PostgreSQL with `pgvector`.
- **Semantic retrieval** — find the most relevant document chunks.
- **Generative AI answers** — Google Gemini generates natural language responses.
- **Extensible** — easily adapt for other vector stores or LLM providers.

---

## ⚙️ Requirements
- **Python** ≥ 3.10
- **PostgreSQL** with `pgvector` extension installed
- **Google Generative AI API key**
- **LangChain** for RAG pipeline

## Enviroment Variables
- DB_HOST='db'
- DB_NAME='db_name'
- DB_USER='db_username'
- DB_PASSWORD='db_password'
- GOOGLE_API_KEY='your_google_api_key'