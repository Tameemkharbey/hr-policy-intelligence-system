# HR Policy Intelligence Assistant

A fully local, offline RAG-based AI assistant for querying internal HR policy documents. Built with LLaMA 3 (via Ollama), FAISS vector search, and LangChain — zero data leaves the local environment.

---

## What It Does

- Answers HR policy questions using semantic retrieval over uploaded PDF documents
- Falls back to general LLM conversation when no policy context is relevant
- Admin role: upload and manage HR policy PDFs dynamically
- Employee role: query policies through a clean chat interface
- Maintains user context (name recognition) across the session

---

## Architecture

```
User Query
    ↓
Streamlit UI
    ↓
LangChain RAG Pipeline
    ├── FAISS vector store (Hugging Face embeddings)
    │       ↑ indexed from HR policy PDFs
    └── LLaMA 3 8B (4-bit quantized) via Ollama
            → fully local inference, no network calls
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3 8B, 4-bit quantized |
| LLM runtime | Ollama (local inference) |
| Vector store | FAISS |
| Embeddings | Sentence-Transformers (Hugging Face) |
| Orchestration | LangChain |
| UI | Streamlit |
| Language | Python |

---

## Setup

```bash
# 1. Clone and create environment
git clone https://github.com/Tameemkharbey/hr-policy-intelligence-system.git
cd hr-policy-intelligence-system
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 2. Install Ollama and pull the model
# https://ollama.com
ollama pull llama3

# 3. Run
python -m streamlit run app.py
```

---

## Features

- **100% offline** — LLaMA 3 runs locally via Ollama, FAISS index stored locally, no external API calls
- **PDF ingestion** — admins upload policy documents; embeddings are generated and stored immediately
- **Role-based access** — separate admin and employee entry points
- **Semantic retrieval** — FAISS cosine similarity search over chunked policy documents
- **Light/dark theme** — Streamlit theme toggle

---

## License

MIT
