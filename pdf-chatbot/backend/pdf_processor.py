from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_pdf(file_path):
    
    #load the pdf file
    loader = PyPDFLoader(file_path)
    doucments = loader.load()
    
    
    #split the doucments into smaller chunks
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    
    text = text_splitter.split_documents(doucments)
    return text

