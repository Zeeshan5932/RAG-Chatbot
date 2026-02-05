from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import uuid
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ============== CONFIG ==============
UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configurable backend URL (change if needed)
BACKEND_URL = "http://localhost:8000"

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot")

# ============== SESSION STATE INITIALIZATION ==============
# Track whether a PDF has been successfully uploaded
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Store chat history as a list of dicts with "role" and "content"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Store the QA chain for answering questions
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Store conversation memory for context
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# ============== PDF PROCESSING FUNCTIONS ==============
def process_pdfs(file_paths):
    """Load and split PDF documents into chunks."""
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


def create_chain(chunks):
    """Create a ConversationalRetrievalChain from PDF chunks."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
        st.stop()

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(api_key=api_key, temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory
    )

# ============== PDF UPLOAD SECTION ==============
if not st.session_state.pdf_uploaded:
    st.subheader("📤 Upload PDF Documents")
    
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process PDFs", key="process_btn"):
        file_paths = []

        with st.spinner("Processing PDFs..."):
            try:
                for file in uploaded_files:
                    path = os.path.join(
                        UPLOAD_DIR,
                        f"{uuid.uuid4()}_{file.name}"
                    )
                    with open(path, "wb") as f:
                        shutil.copyfileobj(file, f)
                    file_paths.append(path)

                # Process PDFs and create QA chain
                chunks = process_pdfs(file_paths)
                st.session_state.qa_chain = create_chain(chunks)

                # Clean up temporary files
                for path in file_paths:
                    os.remove(path)

                # Mark PDF as uploaded and reset chat history
                st.session_state.pdf_uploaded = True
                st.session_state.chat_history = []

                st.success("✅ PDFs processed successfully! Ready to chat.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")

else:
    # ============== CHATBOT INTERFACE ==============
    st.subheader("💬 Chat with Your PDFs")

    # Display chat history
    if st.session_state.chat_history:
        st.write("---")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")
        st.write("---")

    # User input for questions
    user_question = st.text_input(
        "Ask a question about your PDFs:",
        placeholder="Type your question here...",
        key="user_input"
    )

    # Process user question
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Get response from QA chain
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.qa_chain.invoke(
                    {"question": user_question}
                )
                bot_answer = result.get("answer", "Sorry, I couldn't generate an answer.")

                # Add bot response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": bot_answer
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ Error getting answer: {str(e)}")

    # Option to reset and upload new PDFs
    st.divider()
    if st.button("📁 Upload Different PDFs", key="reset_btn"):
        st.session_state.pdf_uploaded = False
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        st.rerun()
