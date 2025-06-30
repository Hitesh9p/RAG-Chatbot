import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if not uploaded_file:
    st.info("📤 Please upload a PDF to begin.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    pdf_path = tmp_file.name

loader = PyPDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the retrieved context to answer."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("system", "Context: {context}")
])

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

rag_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: x["chat_history"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

query = st.chat_input("Ask a question about the PDF")
if query:
    answer = rag_chain.invoke({
        "question": query,
        "chat_history": st.session_state.chat_history
    })
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer))

    st.chat_message("You").write(query)
    st.chat_message("Bot").write(answer)

for msg in st.session_state.chat_history:
    role = "You" if isinstance(msg, HumanMessage) else "Bot"
    st.chat_message(role).write(msg.content)
