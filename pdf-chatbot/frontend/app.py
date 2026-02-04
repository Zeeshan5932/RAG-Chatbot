import streamlit as st
import requests



BACKEND_URL = "http://127.0.0.1:8000"


st.title("PDF Chatbot")
st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading & processing PDF..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(
            f"{BACKEND_URL}/upload-pdf/",
            files={"file": uploaded_file}
        )

    if response.status_code == 200:
        st.success("PDF processed successfully!")

# Ask Question
question = st.text_input("Ask a question from the PDF")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{BACKEND_URL}/ask-question/",
                params={"question": question}
            )
            answer = res.json().get("answer", "No answer")

        st.subheader("Answer")
        st.write(answer)