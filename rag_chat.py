from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

VECTOR_DB_PATH = "vectorstore"

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = Ollama(
        model="llama3:latest",
        temperature=0.1
    )

    print("📄 Leave Policy Chatbot (RAG)")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)


        if not docs:
            print("\nBot: I could not find this information in the policy.\n")
            continue

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an HR assistant.
Answer ONLY using the information provided below.
If the answer is not present, say: "This information is not mentioned in the policy."

Policy:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)
        print("\nBot:", response, "\n")

if __name__ == "__main__":
    main()
