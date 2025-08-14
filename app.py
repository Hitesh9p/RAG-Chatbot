import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Load the API key securely
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except KeyError:
    st.error("Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot — FAISS + HuggingFace + Groq (Cloud Ready)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if not uploaded_file:
    st.info("Please upload a PDF to start.")
    st.stop()

# Save uploaded file to disk before processing
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    tmp_pdf_path = tmp.name

# Load, split, and sanitize PDF pages
loader = PyPDFLoader(tmp_pdf_path)
pages = loader.load()
for p in pages:
    p.page_content = p.page_content or ""
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
for d in chunks:
    d.page_content = d.page_content or ""

# Initialize HF embeddings & FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# Setup Groq LLM & retrieval chain
llm = ChatGroq(api_key=GROQ_API_KEY, model="mixtral-8x7b-32768", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ask and answer
user_query = st.text_input("Ask me anything about your PDF:")
if user_query:
    result = qa({"question": user_query, "chat_history": st.session_state.chat_history})
    
    answer = result.get("answer", "No answer.")
    sources = result.get("source_documents", [])
    
    st.markdown("**Answer:**")
    st.write(answer)
    
    if sources:
        st.markdown("**Sources:**")
        for i, doc in enumerate(sources, start=1):
            content = getattr(doc, "page_content", str(doc))
            st.write(f"{i}. {content[:500]}{'...' if len(content) > 500 else ''}")

    st.session_state.chat_history.append((user_query, answer))

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Conversation:")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
