# # 

# from dotenv import load_dotenv
# load_dotenv()

# from fastapi import FastAPI, UploadFile, File, Query
# import shutil
# import os
# import uuid

# from pdf_processor import process_pdfs
# from qa_chain import create_qa_chain
# from session_manager import SessionManager

# app = FastAPI()

# UPLOAD_DIR = "temp"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# session_manager = SessionManager()

# @app.post("/start-session/")
# def start_session():
#     session_id = str(uuid.uuid4())
#     session_manager.get_session(session_id)
#     return {"session_id": session_id}


# @app.post("/upload-pdfs/")
# async def upload_pdfs(
#     session_id: str = Query(...),
#     files: list[UploadFile] = File(...)
# ):
#     session = session_manager.get_session(session_id)

#     file_paths = []

#     for file in files:
#         path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
#         with open(path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         file_paths.append(path)

#     chunks = process_pdfs(file_paths)

#     session["qa_chain"] = create_qa_chain(
#         chunks,
#         session["memory"]
#     )

#     for path in file_paths:
#         os.remove(path)

#     return {"message": "PDFs uploaded and processed"}


# @app.post("/ask/")
# async def ask_question(
#     session_id: str = Query(...),
#     question: str = Query(...)
# ):
#     session = session_manager.get_session(session_id)

#     if session["qa_chain"] is None:
#         return {"error": "Upload PDFs first"}

#     result = session["qa_chain"].invoke({"question": question})
#     return {"answer": result["answer"]}



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

# ---------------- CONFIG ----------------
UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF Chatbot (All-in-One)")

# ---------------- SESSION INIT ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# ---------------- PDF PROCESSING ----------------
def process_pdfs(file_paths):
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

# ---------------- UI: PDF UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.button("Process PDFs"):
    file_paths = []

    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            path = os.path.join(
                UPLOAD_DIR,
                f"{st.session_state.session_id}_{file.name}"
            )
            with open(path, "wb") as f:
                shutil.copyfileobj(file, f)
            file_paths.append(path)

        chunks = process_pdfs(file_paths)
        st.session_state.qa_chain = create_chain(chunks)

        for path in file_paths:
            os.remove(path)

    st.success("PDFs processed successfully!")

# ---------------- UI: CHAT ----------------
st.divider()
st.subheader("Ask Questions")

question = st.text_input("Ask something from the PDFs")

if st.button("Ask"):
    if st.session_state.qa_chain is None:
        st.warning("Please upload and process PDFs first")
    elif question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke(
                {"question": question}
            )
        st.markdown("### Answer")
        st.write(result["answer"])

# ---------------- CHAT HISTORY ----------------
if st.session_state.memory.chat_memory.messages:
    st.divider()
    st.subheader("Chat History")

    for msg in st.session_state.memory.chat_memory.messages:
        role = "🧑 User" if msg.type == "human" else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg.content}")
