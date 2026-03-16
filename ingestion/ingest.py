from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

documents = []

data_path = "data"

for root, _, files in os.walk(data_path):
    for file in files:
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(root, file))
            documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embedding)

db.save_local("vectorstore")