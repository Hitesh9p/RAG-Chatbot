import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import json
import traceback

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
st.title("📚 RAG Chatbot (Groq API)")
user_query = st.text_input("💬 Ask me something:")

if user_query:
    try:
        # Prepare messages in correct LangChain format
        messages = [
            SystemMessage(content="You are a helpful assistant that answers based on context."),
            HumanMessage(content=user_query)
        ]

        # Debug logging — see exactly what is sent to Groq
        st.subheader("🛠 Debug Log (what is being sent)")
        st.code(json.dumps([{"role": "system", "content": messages[0].content},
                            {"role": "user", "content": messages[1].content}],
                           indent=2), language="json")

        # Call Groq model
        response = llm.invoke(messages)

        # Display the response
        st.subheader("✅ Answer")
        st.write(response.content)

    except Exception as e:
        st.subheader("❌ Error Details")
        st.error(str(e))
        st.text(traceback.format_exc())
