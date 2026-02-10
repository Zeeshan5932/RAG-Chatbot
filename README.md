

# 📄 PDF RAG Chatbot (Streamlit + LangChain)

An **AI-powered PDF chatbot** that allows users to upload one or multiple PDF documents and ask questions about their content.
The system uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers grounded strictly in the uploaded documents.

---

## 🚀 Features

* 📚 Upload **single or multiple PDFs**
* 🤖 Ask natural language questions from PDFs
* 🧠 Uses **RAG (Retrieval-Augmented Generation)**
* 🔍 Embedding-based **semantic search**
* 💬 **Conversation memory** (chat history)
* ⚡ Built with **Streamlit** (all-in-one app)
* 🔐 Secure API key handling using `.env`
* 🧩 No model training required

---

## 🧠 How It Works (RAG Pipeline)

```
PDF Upload
   ↓
Text Extraction (PyPDFLoader)
   ↓
Text Chunking
   ↓
Embeddings (OpenAI)
   ↓
Vector Store (FAISS)
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Chunks + Question
   ↓
LLM (ChatGPT)
   ↓
Final Answer
```

This approach **reduces hallucinations** by forcing the LLM to answer using only the retrieved document context.

---

## 🛠 Tech Stack

| Component   | Technology                         |
| ----------- | ---------------------------------- |
| Language    | Python                             |
| UI          | Streamlit                          |
| LLM         | OpenAI (GPT)                       |
| Framework   | LangChain                          |
| Vector DB   | FAISS                              |
| PDF Parsing | PyPDF                              |
| Memory      | LangChain ConversationBufferMemory |

---

## 📁 Project Structure

```
pdf-chatbot/
│
├── app.py          # Main Streamlit application
├── .env            # OpenAI API key (not committed)
└── README.md
```

---

## 🔑 Prerequisites

* Python 3.9+
* OpenAI API key

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

---

### 2️⃣ Install Dependencies

```bash
pip install streamlit langchain langchain-openai pypdf faiss-cpu python-dotenv tiktoken openai
```

---

### 3️⃣ Create `.env` File

In the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key
```

⚠️ **Do not commit `.env` to GitHub**

---

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 🧪 Usage

1. Upload one or more PDF files
2. Click **Process PDFs**
3. Ask questions related to the document content
4. View answers and chat history

---

## ❗ Important Notes

* The chatbot answers **only from uploaded PDFs**
* No model training is performed
* Each session is isolated using Streamlit session state
* Temporary files are deleted after processing

---

## 🎯 Use Cases

* University / research document Q&A
* Company policy or SOP chatbot
* Legal / financial document analysis
* Resume or proposal review
* Internal knowledge base assistant

---

## 📌 Future Improvements

* 📖 Source citations with page numbers
* 🧠 Support for free/open-source LLMs
* ☁️ Cloud deployment (Docker, AWS, Render)
* 🔐 User authentication
* 🖼️ PDF preview & highlighting

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---




