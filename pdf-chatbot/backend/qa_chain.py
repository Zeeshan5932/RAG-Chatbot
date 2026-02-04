from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


def create_qa_chain(text_chunks):
    # 1️⃣ Embeddings
    embeddings = OpenAIEmbeddings()

    # 2️⃣ Vector Store
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    # 3️⃣ LLM
    llm = ChatOpenAI(temperature=0)

    # 4️⃣ Retrieval QA Chain (✅ CORRECT WAY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    return qa_chain
