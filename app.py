import os
import re
import streamlit as st
import ollama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="wide"
)

# =========================
# Theme State
# =========================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def apply_theme(theme):
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] {
            background-color: #020617;
        }
        .stChatMessage {
            background-color: #020617;
            border-radius: 12px;
            padding: 10px;
        }
        input, textarea {
            background-color: #020617 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }
        section[data-testid="stSidebar"] {
            background-color: #e5e7eb;
        }
        .stChatMessage {
            background-color: white;
            border-radius: 12px;
            padding: 10px;
        }
        input, textarea {
            background-color: white !important;
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)


apply_theme(st.session_state.theme)

# =========================
# Login System
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 HR Assistant Login")

    username = st.text_input("Username")
    role = st.selectbox("Role", ["Employee", "Admin"])

    if st.button("Login"):
        if username.strip():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            st.session_state.messages = []
            st.session_state.user_profile = {}
            st.rerun()
        else:
            st.warning("Please enter a username")

    st.stop()

# =========================
# Load Vector Store
# =========================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================
# Add PDF to Vector DB
# =========================
def add_pdf_to_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    vectorstore.save_local("vectorstore")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### 💼 HR AI Assistant")
    st.markdown(f"👤 **User:** {st.session_state.username}")
    st.markdown(f"🔐 **Role:** {st.session_state.role}")

    st.markdown("---")

    theme_toggle = st.toggle(
        "🌙 Dark Mode",
        value=(st.session_state.theme == "dark")
    )
    st.session_state.theme = "dark" if theme_toggle else "light"
    apply_theme(st.session_state.theme)

    st.markdown("---")
    st.success("🟢 LLaMA 3 Running")
    st.success("🟢 Vector DB Loaded")

    if st.session_state.role == "Admin":
        st.markdown("### 📁 Upload HR Policy PDF")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            add_pdf_to_vectorstore(file_path)
            st.success("✅ Policy added successfully")

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================
# Main Header
# =========================
st.markdown("## 🤖 HR Policy Assistant")
st.caption("Policy-aware chatbot + general AI — fully local & secure")

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# =========================
# Display Chat History
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat Input
# =========================
user_query = st.chat_input("Ask HR policy questions or chat freely...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Memory: Name
    name_match = re.search(r"(my name is|i am|call me)\s+([a-zA-Z]+)", user_query.lower())
    if name_match:
        name = name_match.group(2).capitalize()
        st.session_state.user_profile["name"] = name
        bot_reply = f"Nice to meet you, **{name}** 😊"
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.stop()

    docs = retriever.invoke(user_query)

    policy_keywords = [
        "leave", "policy", "salary", "pay", "office",
        "working hours", "attendance", "vacation", "sick"
    ]

    is_policy_question = any(k in user_query.lower() for k in policy_keywords)

    if docs and is_policy_question:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an HR policy assistant.
Answer strictly using the policy below.
If the answer is missing, say:
"This information is not mentioned in the HR policy."

HR Policy:
{context}

Question:
{user_query}
"""

        response = ollama.chat(
            model="llama3:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        bot_reply = response["message"]["content"]

    else:
        name = st.session_state.user_profile.get("name", "there")

        response = ollama.chat(
            model="llama3:latest",
            messages=[
                {"role": "system", "content": f"You are chatting with {name}."}
            ] + st.session_state.messages
        )
        bot_reply = response["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
