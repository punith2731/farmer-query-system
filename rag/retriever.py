from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_retriever():
    vectorstore_path = Path("vectorstore")
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            "Vector store not found. Run ingestion/ingest.py first to build it."
        )

    db = FAISS.load_local(
        str(vectorstore_path),
        embedding,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )