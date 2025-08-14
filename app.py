# ---------------- FIX FOR SQLITE3 ----------------
# This must be at the very top of your script
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -------------------------------------------------

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import tempfile
import traceback
import json
import os # It's good practice to use os.path.join

# ---------------- CONFIG ----------------
MODEL_NAME = "llama3-70b-8192"

# ---------------- API KEY HANDLING ----------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ Missing `GROQ_API_KEY` in Streamlit secrets.")
    st.info("Please add it to your secrets.toml file and restart the app.")
    st.stop()

# ---------------- LLM INITIALIZATION ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

# ---------------- STREAMLIT UI ----------------
st.title("📚 RAG Chatbot with PDF + Groq API")
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

# Store vector DB in session
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file:
    with st.spinner("Processing PDF... this might take a moment ⏳"):
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load and split PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # ✅ FREE local embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Store in Chroma using the corrected sqlite3
            vectorstore = Chroma.from_documents(splits, embedding=embeddings)
            st.session_state.vectorstore = vectorstore

            # Clean up the temporary file
            os.remove(tmp_path)

            st.success("✅ PDF processed and ready for questions.")

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.text(traceback.format_exc())

# Question input
user_query = st.text_input("💬 Ask something from the document:")

if user_query and st.session_state.vectorstore:
    try:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        with st.spinner("Finding answer..."):
            result = qa_chain.invoke({"query": user_query}) # Use invoke for newer LangChain versions

        # Show answer
        st.subheader("✅ Answer")
        st.write(result["result"])

        # Show sources
        st.subheader("📄 Sources")
        for doc in result["source_documents"]:
            source = doc.metadata.get('source', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            st.info(f"Source: {os.path.basename(source)}, Page: {page + 1}")
            st.write(doc.page_content[:300] + "...")


    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        st.text(traceback.format_exc())
elif user_query:
    st.warning("⚠ Please upload and process a PDF first.")
