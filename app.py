import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ---------------------------
# Load API key from secrets
# ---------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📄 RAG Chatbot with Groq")
st.caption("Upload a PDF and ask questions. Powered by FAISS + Groq LLM.")

uploaded_file = st.file_uploader("📤 Upload a PDF", type=["pdf"])
user_question = st.text_input("💬 Ask a question about the PDF:")

# ---------------------------
# Process PDF and Build Vectorstore
# ---------------------------
if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

    # ---------------------------
    # Chat with Groq
    # ---------------------------
    if user_question:
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(user_question)

        # Create context from retrieved docs
        context = "\n\n".join([d.page_content for d in retrieved_docs])

        llm = ChatGroq(groq_api_key=groq_api_key, model="mixtral-8x7b-32768")

        prompt = f"""
        You are a helpful assistant. 
        Use the following context to answer the user's question. 
        If the answer is not in the context, say "I couldn't find that in the document."

        Context:
        {context}

        Question:
        {user_question}
        """

        response = llm.invoke(prompt)

        # Display answer
        st.subheader("📝 Answer:")
        st.write(response.content)
