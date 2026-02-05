# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA


# def create_qa_chain(text_chunks):
#     # 1️⃣ Embeddings
#     embeddings = OpenAIEmbeddings()

#     # 2️⃣ Vector Store
#     vectorstore = FAISS.from_documents(text_chunks, embeddings)

#     # 3️⃣ LLM
#     llm = ChatOpenAI(temperature=0)

#     # 4️⃣ Retrieval QA Chain (✅ CORRECT WAY)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         chain_type="stuff"
#     )

#     return qa_chain


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


def create_qa_chain(chunks, memory):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa_chain
