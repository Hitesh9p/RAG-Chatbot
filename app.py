# app.py
import os
import tempfile
import textwrap
import streamlit as st

# Flexible imports with helpful errors if versions differ
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    from langchain.document_loaders import PyPDFLoader  # fallback

try:
    # newer langchain location
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings + vectorstore imports (try both community and core locations)
HuggingFaceEmbeddings = None
FAISS = None
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        pass
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        pass

# Chain and LLM
try:
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    ConversationalRetrievalChain = None

# Groq LLM
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# Page layout
st.set_page_config(page_title="RAG PDF Chatbot (FAISS + HF)", layout="wide")
st.title("📄 RAG PDF Chatbot — FAISS + Hugging Face Embeddings + Groq")

# Get GROQ API key: prefer st.secrets, fallback to input
groq_key = st.secrets.get("GROQ_API_KEY") if st.secrets and "GROQ_API_KEY" in st.secrets else None
groq_key_input = st.text_input("GROQ API Key (or set in Streamlit Secrets)", type="password", value="" if groq_key else "")
if not groq_key:
    groq_key = groq_key_input.strip() or None

if not groq_key:
    st.info("Enter your GROQ API key (or add `GROQ_API_KEY` to Streamlit secrets) — the app will still let you upload and process a PDF, but you won't be able to generate answers until the key is provided.")
# Show current key source for debugging (don't print the key)
if groq_key:
    st.success("GROQ API key detected (source: secrets or input).")

# Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Only continue when a file is uploaded
if uploaded_file is None:
    st.info("Upload a PDF to process.")
    st.stop()

# Process PDF to temp file (PyPDFLoader expects a file path)
try:
    with st.spinner("Saving uploaded PDF to a temporary file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
except Exception as e:
    st.error(f"Failed to save uploaded file: {e}")
    st.stop()

# Load PDF pages
try:
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
except Exception as e:
    st.error(f"Error loading PDF with PyPDFLoader:\n{e}")
    st.stop()

# Ensure page_content is a string
for p in pages:
    if not isinstance(p.page_content, str):
        p.page_content = str(p.page_content or "")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)

# Ensure chunk contents are strings
for d in chunks:
    if not isinstance(d.page_content, str):
        d.page_content = str(d.page_content or "")

# Embeddings: instantiate HuggingFaceEmbeddings
if HuggingFaceEmbeddings is None:
    st.error("HuggingFaceEmbeddings import failed. Make sure `sentence-transformers` is in requirements.")
    st.stop()

try:
    # model name recommended: sentence-transformers/all-MiniLM-L6-v2
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
    st.stop()

# Vectorstore: FAISS
if FAISS is None:
    st.error("FAISS import failed. Make sure `faiss-cpu` is in requirements.")
    st.stop()

try:
    vectorstore = FAISS.from_documents(chunks, hf_embeddings)
except Exception as e:
    st.error(f"Failed to build FAISS vectorstore: {e}")
    st.stop()

st.success("PDF processed and indexed into FAISS ✅")

# Prepare retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# chat history list of (q,a)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Query input
query = st.text_input("Ask a question about the PDF (you can ask follow-ups):").strip()

if query:
    # If no groq key, show friendly error and stop before LLM call
    if not groq_key:
        st.error("GROQ API key required to generate answers. Add it to Streamlit Secrets or paste it above.")
        st.stop()

    # Build LLM
    if ChatGroq is None:
        st.error("ChatGroq import failed. Ensure `langchain_groq` is installed and available.")
        st.stop()

    try:
        # instantiate ChatGroq with common parameter names (works with multiple versions)
        try:
            llm = ChatGroq(api_key=groq_key, model="mixtral-8x7b-32768", temperature=0)
        except TypeError:
            # alternate constructor names
            llm = ChatGroq(groq_api_key=groq_key, model_name="mixtral-8x7b-32768", temperature=0)
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq LLM: {e}")
        st.stop()

    # Try using ConversationalRetrievalChain if available for nicer behavior
    answer = None
    sources = None
    if ConversationalRetrievalChain is not None:
        try:
            rag = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
            # prefer dict-style call (works with many langchain versions)
            try:
                output = rag({"question": query, "chat_history": st.session_state.chat_history})
            except Exception:
                # fallback to run() which returns text only in older versions
                output_text = rag.run(query)
                output = {"answer": output_text, "source_documents": None}
            # extract answer and sources flexibly
            if isinstance(output, dict):
                answer = output.get("answer") or output.get("result") or output.get("output_text") or str(output)
                sources = output.get("source_documents")
            else:
                answer = str(output)
                sources = None
        except Exception as e:
            # Show the concrete error so we can debug rather than a redacted crash
            st.error(f"ConversationalRetrievalChain failed: {e}")
    # If chain didn't produce an answer, fall back to manual prompt + LLM call
    if not answer:
        st.info("Falling back to manual prompt approach.")
        # fetch relevant docs
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            try:
                docs = retriever.get_relevant_documents(query)  # try again (some versions differ)
            except Exception as e:
                st.error(f"Retriever failed to return documents: {e}")
                st.stop()

        # build a context string from top docs (limit to first N chars)
        context_texts = []
        for d in docs[:5]:
            text = d.page_content if hasattr(d, "page_content") else str(d)
            # keep it short per doc to avoid overlong prompt
            context_texts.append(textwrap.shorten(text, width=1200, placeholder=" ..."))

        context = "\n\n---\n\n".join(context_texts)
        system_prompt = "You are a helpful assistant. Use the context below to accurately answer the question. If the context doesn't contain the answer, say you don't know."
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Call llm - many LangChain LLMs implement __call__ or generate; be defensive
        try:
            # preferred: call like an LLM returning a string
            generated = llm(full_prompt)
            # if generated is a dict-like or special object, try to extract text
            if isinstance(generated, dict):
                answer = generated.get("text") or generated.get("content") or str(generated)
            else:
                answer = str(generated)
        except TypeError:
            # fallback to generate / predict if available
            try:
                generated = llm.generate([full_prompt])
                # try to extract text
                answer = str(generated)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                st.stop()
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

        sources = docs

    # Save & display chat
    st.session_state.chat_history.append((query, answer))
    st.subheader("Answer:")
    st.write(answer)

    if sources:
        st.subheader("Top source snippets:")
        for i, doc in enumerate(sources[:5]):
            snippet = doc.page_content if hasattr(doc, "page_content") else str(doc)
            st.markdown(f"**Source {i+1}:**")
            st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))

# Show chat history below
if st.session_state.get("chat_history"):
    st.markdown("---")
    st.subheader("Conversation history")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
