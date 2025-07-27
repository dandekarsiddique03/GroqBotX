# GroqBot X – Chat with Your Documents, Smarter & Faster!

[![Streamlit App](https://groqbotx.streamlit.app/)

GroqBot X is an AI-powered document interaction system that allows you to **chat with your PDF, DOCX, or TXT documents** in real-time. It uses advanced NLP techniques, semantic search, and retrieval-augmented generation (RAG) to help you get instant, contextual answers from your documents with high accuracy.

---

## 🚀 Live Demo

👉 Try the live app: **[https://groqbotx.streamlit.app/](https://groqbotx.streamlit.app/)**

---

## 📂 Features

- 📄 **Multi-format Support**: Upload PDF, DOCX, or TXT files.
- 🔁 **Automatic Conversion**: DOCX and TXT files are converted to PDF for consistency.
- 🧠 **Semantic Search**: Embedding-based retrieval using HuggingFace + FAISS.
- 💬 **Real-time Chat**: Ask questions and get context-based answers using Groq's LLaMA3 LLM via LangChain.
- 📎 **Source Transparency**: Shows the most relevant document chunks used to answer your query.
- 🔒 **Privacy-Focused**: Local embedding & retrieval, no data sent to external servers.
- 🧹 **Clear All**: Easily delete files and reset the session with one click.

---

## 🧰 Tech Stack

| Component              | Technology                                 |
|------------------------|--------------------------------------------|
| Frontend               | Streamlit                                  |
| Language               | Python                                     |
| Vector DB              | FAISS                                      |
| Embedding Model        | HuggingFace (MiniLM-L6-v2)                 |
| LLM                    | Groq API (LLaMA3-8B) via LangChain         |
| Document Loaders       | PyMuPDF, python-docx, chardet              |
| PDF Conversion         | ReportLab                                  |
| Environment Mgmt       | Python-dotenv                              |

---

## 🧠 How It Works

1. Upload PDF, DOCX, or TXT documents.
2. Files are standardized to PDF and split into text chunks.
3. Chunks are converted to vector embeddings using HuggingFace.
4. FAISS stores these embeddings for fast similarity search.
5. When a question is asked, relevant chunks are retrieved.
6. Groq’s LLaMA3 generates a context-aware answer using LangChain.
7. Output is shown along with the document chunks used.

---

## 🧪 Setup Locally

1. Clone this repo  
   ```bash
   git clone https://github.com/your-username/groqbotx.git
   cd groqbotx
