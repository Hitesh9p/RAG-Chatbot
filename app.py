# app.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# ---- Page config ----
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot with FAISS + HuggingFace + Groq")

# ---- Initialize session state ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- File uploader ----
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading & processing PDF..."):
        # Load PDF
        loader = PyPDFLoader(uploaded_file)
        pages = loader.load()

        # Ensure page content is always a string
        for p in pages:
            if not isinstance(p.page_content, str):
                p.page_content = str(p.page_content or "")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(pages)

        # Ensure every chunk is a string
        for doc in split_docs:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content or "")

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create FAISS vectorstore
        st.session_state.vectorstore = FAISS.from_documents(split_docs, embeddings)

    st.success("✅ PDF processed successfully!")

# ---- Chat input ----
query = st.text_input("Ask a question about your PDF:")

if query and st.session_state.vectorstore is not None:
    with st.spinner("Generating answer..."):
        retriever = st.session_state.vectorstore.as_retriever()

        # Groq LLM
        llm = ChatGroq(
            model="llama3-8b-8192",  # or another Groq-supported model
            temperature=0,
            api_key=st.secrets["GROQ_API_KEY"]  # store in Streamlit secrets
        )

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Ensure query is string
        query = str(query).strip()

        result = rag_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        # Save chat history
        st.session_state.chat_history.append((query, result["answer"]))

        # Display answer
        st.markdown("**Answer:**")
        st.write(result["answer"])

        # Show sources
        if result.get("source_documents"):
            with st.expander("📚 Sources"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content[:500] + "...")

elif query and st.session_state.vectorstore is None:
    st.warning("⚠ Please upload and process a PDF first.")
