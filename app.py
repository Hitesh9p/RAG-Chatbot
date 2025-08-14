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

# ---------------- CONFIG ----------------
MODEL_NAME = "llama3-70b-8192"

# ---------------- API KEY HANDLING ----------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ Missing `GROQ_API_KEY` in Streamlit secrets.")
    st.stop()

# ---------------- LLM INITIALIZATION ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

# ---------------- STREAMLIT UI ----------------
st.title("📚 RAG Chatbot with PDF + Groq API (Free Embeddings)")
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

# Store vector DB in session
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file:
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
        
        # Store in Chroma
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        st.session_state.vectorstore = vectorstore

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

        # Debug log
        st.subheader("🛠 Debug Log")
        st.code(json.dumps({"query": user_query}, indent=2), language="json")

        result = qa_chain({"query": user_query})

        # Show answer
        st.subheader("✅ Answer")
        st.write(result["result"])

        # Show sources
        st.subheader("📄 Sources")
        for doc in result["source_documents"]:
            st.write(doc.metadata, doc.page_content[:200] + "...")

    except Exception as e:
        st.error(f"Error during query: {str(e)}")
        st.text(traceback.format_exc())
elif user_query:
    st.warning("⚠ Please upload and process a PDF first.")
