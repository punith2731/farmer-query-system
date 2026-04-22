import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embedding_model

embedding = get_embedding_model()


def _validate_embedding_dimension(db: FAISS) -> None:
    """Raise a helpful error when vectorstore and embedding dims do not match."""
    # FAISS index dimension
    index_dim = int(getattr(db.index, "d", 0) or 0)

    # Embedding dimension for current model
    probe_vector = embedding.embed_query("dimension_probe")
    embed_dim = len(probe_vector)

    if index_dim and embed_dim and index_dim != embed_dim:
        raise ValueError(
            "Embedding dimension mismatch: vectorstore dimension "
            f"({index_dim}) != current embedding dimension ({embed_dim}). "
            "Rebuild the vectorstore using the same EMBEDDING_MODEL by running ingestion/ingest.py."
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

    _validate_embedding_dimension(db)

    requested_k = os.getenv("RETRIEVER_TOP_K", "15").strip()
    try:
        top_k = int(requested_k)
    except Exception:
        top_k = 15
    top_k = max(10, min(20, top_k))

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )