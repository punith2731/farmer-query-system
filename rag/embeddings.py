import os

from dotenv import load_dotenv
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]
from langchain_openai import OpenAIEmbeddings

load_dotenv()

_MODEL_ALIASES = {
    "text-embedding-3-large": "text-embedding-3-large",
    "bge-large-en": "BAAI/bge-large-en-v1.5",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def _normalize_choice(value: str) -> str:
    return (value or "").strip().lower()


def _resolve_model(choice: str) -> str:
    normalized = _normalize_choice(choice)
    if normalized in _MODEL_ALIASES:
        return _MODEL_ALIASES[normalized]
    return choice


def get_embedding_model():
    """
    Returns a configured embedding model for both ingestion and retrieval.

    Supported options via EMBEDDING_MODEL:
    - text-embedding-3-large (OpenAI)
    - bge-large-en (mapped to BAAI/bge-large-en-v1.5)
    - all-MiniLM-L6-v2 (mapped to sentence-transformers/all-MiniLM-L6-v2)
    """
    requested = os.getenv("EMBEDDING_MODEL", "bge-large-en").strip()
    normalized = _normalize_choice(requested)

    if normalized == "text-embedding-3-large":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_MODEL=text-embedding-3-large"
            )
        return OpenAIEmbeddings(model="text-embedding-3-large")

    model_name = _resolve_model(requested)
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
