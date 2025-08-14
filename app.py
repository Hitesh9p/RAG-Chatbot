import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# -------------------- CONFIG --------------------
st.set_page_config(page_title="RAG Chatbot with Groq", layout="wide")

# Ensure Groq API key is set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 Please set your Groq API key in the environment variable GROQ_API_KEY.")
    st.stop()

# -------------------- UI --------------------
st.title("📄 PDF RAG Chatbot (Groq + LangChain)")
st.write("Upload a PDF, ask questions, and get AI-powered answers.")

uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

# -------------------- PROCESS PDF --------------------
if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    st.success(f"✅ Loaded {len(documents)} pages from PDF.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embeddings)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # -------------------- CHATBOT --------------------
    llm = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

    # Custom prompt
    template = """
    You are a helpful assistant answering questions based on the provided document.
    Use only the context from the document.
    If the answer is not in the document, say "I couldn't find that in the document."

    Context: {context}
    Question: {question}
    Answer:
    """
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_query = st.text_input("💬 Ask a question from the PDF:")

    if user_query:
        try:
            result = qa(
                {"question": user_query, "chat_history": st.session_state.chat_history}
            )

            st.session_state.chat_history.append((user_query, result["answer"]))

            # Display answer
            st.markdown(f"**🤖 Answer:** {result['answer']}")

            # Show sources
            with st.expander("📄 Sources"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
