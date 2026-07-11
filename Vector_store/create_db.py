from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load PDF
import os

loader = PyPDFLoader(
    os.path.join("Document Loader", "deeplearning.pdf")
)
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

print("Chunks:", len(chunks))

# LOCAL embeddings (NO API, NO LIMITS)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Database created successfully!")