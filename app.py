import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot with Groq API")

# API key input
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = st.text_input(
        "Enter your Groq API key", type="password"
    )

if not st.session_state.GROQ_API_KEY:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Store in FAISS
        vectorstore = FAISS.from_documents(docs, embeddings)

        # RAG chain
        llm = ChatGroq(
            groq_api_key=st.session_state.GROQ_API_KEY,
            model_name="mixtral-8x7b-32768"
        )

        retriever = vectorstore.as_retriever()
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever
        )

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User query
        query = st.text_input("Ask something about your PDF:")
        if query:
            with st.spinner("Generating answer..."):
                result = rag_chain.invoke({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })
                st.session_state.chat_history.append((query, result["answer"]))
                st.markdown(f"**Answer:** {result['answer']}")

        # Show chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
