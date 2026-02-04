from fastapi import FastAPI, UploadFile, File,Query
import shutil
import os

from pdf_processor import process_pdf
from qa_chain import create_qa_chain
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

qa_chain = None

UPLOAD_DIR = "temp"   # 👈 temp folder ka naam


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain

    # ✅ agar temp folder nahi hai to bana do
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # file ka full path
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    # file save karo
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # PDF process karo
    text_chunks = process_pdf(file_location)

    # QA chain banao
    qa_chain = create_qa_chain(text_chunks)

    # optional: file delete kar do
    os.remove(file_location)

    return {"message": "PDF processed and QA chain created successfully."}


@app.post("/ask-question/")
async def ask_question(question: str = Query(...)):
    global qa_chain

    if qa_chain is None:
        return {"error": "Please upload a PDF first."}

    answer = qa_chain.run(question)
    return {"answer": answer}
