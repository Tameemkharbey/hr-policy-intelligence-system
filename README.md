# 🤖 HR Policy Assistant (RAG + LLaMA 3)

A fully **local**, **secure**, and **intelligent HR chatbot** built using **LLaMA 3**, **Ollama**, **LangChain**, **FAISS**, and **Streamlit**.

This system can:
- Answer HR policy questions using **RAG**
- Chat normally like a general AI assistant
- Remember user details (e.g., name)
- Support Admin & Employee roles
- Run **100% locally** (no OpenAI / cloud APIs)

---

## 🚀 Features

✅ Local LLaMA 3 (Ollama)  
✅ Retrieval-Augmented Generation (FAISS)  
✅ HR Policy PDF Upload (Admin)  
✅ Role-based Login System  
✅ User Memory (Name recognition)  
✅ Dark / Light Mode Toggle 🌙☀️  
✅ Fully Offline & Secure  
✅ Streamlit UI Dashboard  

---

## 🧠 Architecture


---

## 🛠️ Tech Stack

| Component | Technology |
|--------|-----------|
| LLM | LLaMA 3 (8B, 4-bit) |
| LLM Runtime | Ollama |
| Vector DB | FAISS |
| Embeddings | Sentence-Transformers |
| Framework | LangChain |
| UI | Streamlit |
| Language | Python |


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/hr-policy-assistant.git
cd hr-policy-assistant

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install & Run Ollama

Download Ollama from:
👉 https://ollama.com

Pull LLaMA 3:

ollama pull llama3


Verify:

ollama list

▶️ Run the Application
python -m streamlit run app.py


