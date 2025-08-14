# 📚 RAG Chatbot with PDF, Groq, and Free Embeddings

This project is a web application built with Streamlit that allows you to chat with your PDF documents. It uses the Retrieval-Augmented Generation (RAG) technique to provide accurate answers based on the content of the uploaded PDF.

The application leverages the high-speed inference of the Groq API for language model processing and uses free, locally-run Hugging Face sentence-transformer models for document embeddings.

<img width="1440" height="777" alt="Screenshot 2025-08-14 at 7 07 52 PM" src="https://github.com/user-attachments/assets/febbaa65-9768-4695-97ec-47b7c7789329" />


---

## 🚀 Overview

The core functionality of this application is to make any PDF document interactive. A user can upload a PDF, and the system will process and "learn" its content. Afterward, the user can ask questions in natural language, and the chatbot will retrieve relevant information from the document to generate a coherent and contextually accurate answer. This eliminates the need to manually search through lengthy documents.

---

## ✨ Features

- **PDF Upload**: Easily upload any PDF file through a simple web interface.
- **Document Processing**: Automatically splits the document into manageable chunks for efficient processing.
- **Local Embeddings**: Uses the `all-MiniLM-L6-v2` model from Hugging Face to create vector embeddings for free, without needing a paid API.
- **Vector Storage**: Employs ChromaDB to store the document embeddings in a searchable vector database.
- **High-Speed Q&A**: Integrates with the Groq API (using the Llama 3 model) for near-instantaneous answer generation.
- **Source Citing**: Displays the specific parts of the source document that were used to formulate the answer, ensuring transparency and trust.
- **Self-Contained & Deployable**: Designed to be easily deployed on platforms like Streamlit Community Cloud.

---

## 🛠️ Tech Stack

- **Backend / Web Framework**: [Streamlit](https://streamlit.io/)
- **LLM API**: [Groq](https://groq.com/)
- **Core LLM Orchestration**: [LangChain](https://www.langchain.com/)
- **Embeddings Model**: [Hugging Face `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **PDF Processing**: `PyPDFLoader`

---

## ⚙️ Setup and Installation

To run this project locally, follow these steps.

### 1. Clone the Repository

```bash
git clone https://github.com/Hitesh9p/RAG-Chatbot/
cd your-repo-name
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content and install the packages.

**`requirements.txt`**:
```txt
streamlit
langchain-groq
langchain
langchain-community
langchain-text-splitters
huggingface-hub
sentence-transformers
pypdf
# Fix for ChromaDB's sqlite3 dependency on Streamlit Cloud
pysqlite3-binary
chromadb==0.4.24
# Fix for numpy 2.0 incompatibility with chromadb<0.5
numpy==1.26.4
```

Then, run the installation command:
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

The application requires a Groq API key. Create a file named `.streamlit/secrets.toml` and add your key.

**`.streamlit/secrets.toml`**:
```toml
GROQ_API_KEY = "gsk_YourActualApiKeyGoesHere"
```

---

## ▶️ How to Run

Once the setup is complete, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to use the application.

---

## ☁️ Deployment on Streamlit Cloud

This app is configured for easy deployment on Streamlit Community Cloud. However, due to system dependencies of the `ChromaDB` library, you need to follow a specific setup.

### 1. `requirements.txt`

Ensure your `requirements.txt` file is exactly as specified in the installation section. The pinned versions of `pysqlite3-binary`, `chromadb`, and `numpy` are crucial for a successful deployment.

### 2. `app.py` Fix

Make sure the following lines are at the **very top** of your `app.py` script. This forces the app to use the compatible `sqlite3` library packaged with `pysqlite3-binary`.

```python
# FIX FOR SQLITE3
# This must be at the very top of your script
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### 3. Deploy

Connect your GitHub repository to your Streamlit Cloud account and deploy the application. The platform will automatically install the requirements and run the app.

---

## 🚨 Troubleshooting

Here are solutions to common errors encountered during development and deployment:

1.  **Error: `Your system has an unsupported version of sqlite3`**
    -   **Cause**: The default environment on Streamlit Cloud has an outdated version of `sqlite3`, which is incompatible with ChromaDB.
    -   **Solution**: Add `pysqlite3-binary` to `requirements.txt` and the monkey-patch code snippet to the top of `app.py` as described in the deployment section.

2.  **Error: `AttributeError: np.float_ was removed in the NumPy 2.0 release`**
    -   **Cause**: `chromadb==0.4.24` is not compatible with `numpy>=2.0`.
    -   **Solution**: Pin the NumPy version in your `requirements.txt` file by adding the line: `numpy==1.26.4`.

3.  **Error: `groq.AuthenticationError: Error code: 401 - Invalid API Key`**
    -   **Cause**: The `GROQ_API_KEY` in your Streamlit secrets is either missing, incorrect, or has expired.
    -   **Solution**:
        1.  Go to the [GroqCloud Console](https://console.groq.com/keys) and generate a new API key.
        2.  In your Streamlit Cloud app settings, go to **Secrets** and ensure your `secrets.toml` file contains the correct key.
        3.  Reboot the application after saving the new secret.

---

## 📄 License

Free to use
---

## 🙏 Acknowledgements

-   Thanks to the teams behind [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and [Groq](https://groq.com/) for their incredible tools that make projects like this possible.
-   The [Hugging Face](https://huggingface.co/) community for providing open-source models.
