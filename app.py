import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 Chat with your PDF (Groq + FAISS)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # ✅ Free cloud-compatible embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="mixtral-8x7b-32768"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    user_question = st.text_input("Ask something about your PDF:")

    if user_question:
        try:
            result = qa({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })

            answer = result.get("answer", "No answer found.")
            st.markdown(f"**Answer:** {answer}")

            sources = result.get("source_documents", [])
            if sources:
                st.markdown("### Sources:")
                for idx, doc in enumerate(sources, start=1):
                    content = getattr(doc, "page_content", str(doc))
                    st.write(f"**Source {idx}:** {content}")

            st.session_state.chat_history.append((user_question, answer))

        except Exception as e:
            st.error(f"Error processing question: {e}")

else:
    st.info("Please upload a PDF to begin.")
