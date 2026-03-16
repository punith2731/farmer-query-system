import os
import re

from dotenv import load_dotenv
from langchain_openai import OpenAI
from rag.retriever import get_retriever

load_dotenv()

INVALID_QUERY_RESPONSE = "this content is not from our farmer query system"

_FARMER_DOMAIN_KEYWORDS = {
    "agri", "agriculture", "farmer", "farming", "crop", "crops", "seed", "seeds",
    "soil", "fertilizer", "fertiliser", "manure", "pesticide", "pest", "disease",
    "irrigation", "drip", "sprinkler", "harvest", "yield", "weed", "insect",
    "maize", "corn", "paddy", "rice", "wheat", "cotton", "tomato", "chilli",
    "sugarcane", "onion", "potato", "millet", "groundnut", "soybean", "dal",
    "horticulture", "livestock", "cattle", "dairy", "goat", "sheep", "poultry",
    "weather", "rain", "monsoon", "acre", "hectare", "nursery", "vermicompost",
    "leaf", "leaves", "stem", "root", "yellow", "wilting", "blight", "rot",
    "pm-kisan", "kisan", "subsidy", "scheme", "loan", "insurance", "fpo",
}

_NON_FARMER_HINTS = {
    "joke", "movie", "song", "cricket", "football", "ipl", "netflix",
    "bitcoin", "celebrity", "politics", "programming", "python code", "java code",
    "stock market", "share market", "relationship", "birthday wish", "poem",
}

_SMALL_TALK_EXACT = {
    "hi", "hello", "hey", "how are you", "who are you", "what is your name",
    "good morning", "good evening", "good night",
}

def _format_context(docs):
    if not docs:
        return "No relevant context was found in the knowledge base."

    return "\n\n".join(doc.page_content for doc in docs)


def _retrieve_documents(query, retriever):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)

    # Backward compatibility for older retriever interfaces.
    return retriever.get_relevant_documents(query)


def _is_farmer_domain_query(query):
    normalized = (query or "").strip().lower()
    if not normalized:
        return False

    return any(keyword in normalized for keyword in _FARMER_DOMAIN_KEYWORDS)


def _is_clearly_non_farmer_query(query):
    normalized = " ".join((query or "").strip().lower().split())
    if not normalized:
        return True

    if normalized in _SMALL_TALK_EXACT:
        return True

    return any(hint in normalized for hint in _NON_FARMER_HINTS)


def _build_prompt(query, context, response_language="English"):
    language_instruction = (
        "Write the final answer in Kannada (ಕನ್ನಡ) using natural, farmer-friendly wording. "
        "Keep crop names, scientific terms, and units (kg/acre, ml/L, NPK) clear and accurate."
        if (response_language or "").strip().lower() == "kannada"
        else "Write the final answer in English."
    )

    return (
        "You are an agriculture advisory assistant. Use only the given context from PDF documents to answer. "
        "Do not add facts outside the provided context. "
        "Write the answer in clean, grammatically correct, easy-to-read sentences. "
        f"{language_instruction} "
        "If the context is insufficient, say: 'I could not find enough information in the uploaded documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        return None

    return api_key


def _is_quota_error(exc):
    message = str(exc).lower()
    return (
        "insufficient_quota" in message
        or "error code: 429" in message
        or "you exceeded your current quota" in message
    )


def _normalize_text(text):
    cleaned = (text or "")

    # Remove common PDF/OCR artifacts.
    cleaned = cleaned.replace("�", "")
    cleaned = cleaned.replace("\u00ad", "")  # soft hyphen

    # Normalize common nutrient notation formatting.
    cleaned = re.sub(r"\bP\s*2\s*O\s*5\b", "P2O5", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bK\s*2\s*O\b", "K2O", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bP\s*O\b", "P2O5", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bK\s*O\b", "K2O", cleaned, flags=re.IGNORECASE)

    # Fix frequent OCR misspellings.
    cleaned = re.sub(r"\beficiency\b", "efficiency", cleaned, flags=re.IGNORECASE)

    # Normalize fertilizer ratio patterns like 40: 20: 20 -> 40:20:20.
    cleaned = re.sub(r"\b(\d+)\s*:\s*(\d+)\s*:\s*(\d+)\b", r"\1:\2:\3", cleaned)

    # Remove stray OCR residue near nutrient notation.
    cleaned = re.sub(
        r"K2O\s+per\s+hectare\s+2\s+5\s+2\s+respectively",
        "K2O per hectare respectively",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Clean spacing around punctuation/symbols.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;!?])(\S)", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    cleaned = re.sub(r"\b(\d+)\s*:\s*(\d+)\s*:\s*(\d+)\b", r"\1:\2:\3", cleaned)
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace(" :", ":")

    return cleaned.strip()


def _clean_answer_text(text):
    cleaned = _normalize_text(text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    return cleaned


def _extract_query_keywords(query):
    stopwords = {
        "the", "is", "are", "am", "what", "when", "where", "why", "how", "to",
        "for", "of", "and", "or", "in", "on", "my", "me", "a", "an", "do", "i",
        "can", "with", "after", "before", "should", "would", "could",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", (query or "").lower())
    return {token for token in tokens if token not in stopwords and len(token) > 2}


def _split_sentences(text):
    normalized = _normalize_text(text)
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [_clean_answer_text(p.strip(" -•\t")) for p in parts if p.strip()]


def _select_relevant_sentences(query, docs, max_sentences=4):
    keywords = _extract_query_keywords(query)
    selected = []
    seen = set()

    for doc in docs[:5]:
        for sentence in _split_sentences(doc.page_content):
            sentence_lower = sentence.lower()
            if keywords:
                if not any(keyword in sentence_lower for keyword in keywords):
                    continue
            if len(sentence) < 30:
                continue

            dedupe_key = sentence.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            selected.append(sentence)
            if len(selected) >= max_sentences:
                return selected

    # Fallback: if keyword filtering found nothing, take the first clean sentences.
    if not selected:
        for doc in docs[:3]:
            for sentence in _split_sentences(doc.page_content):
                if len(sentence) >= 30:
                    dedupe_key = sentence.lower()
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    selected.append(sentence)
                if len(selected) >= max_sentences:
                    return selected

    return selected


def _retrieval_only_fallback_answer(docs, query=""):
    if not docs:
        return "No relevant context was found in the local knowledge base."

    selected_sentences = _select_relevant_sentences(query, docs)
    if not selected_sentences:
        return "I could not find enough information in the uploaded documents."

    answer = " ".join(selected_sentences)
    return _clean_answer_text(answer)

def ask(query, response_language="English"):
    try:
        # Be permissive for farmer users: reject only clearly non-farmer queries.
        if _is_clearly_non_farmer_query(query) and not _is_farmer_domain_query(query):
            return INVALID_QUERY_RESPONSE

        api_key = _get_openai_api_key()
        if not api_key:
            return (
                "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file "
                "(replace 'your_openai_api_key_here' with your real key) and restart the app."
            )

        retriever = get_retriever()
        docs = _retrieve_documents(query, retriever)
        context = _format_context(docs)
        llm = OpenAI(temperature=0)
        response = llm.invoke(_build_prompt(query, context, response_language=response_language))

        if isinstance(response, str):
            return _clean_answer_text(response)

        # Some LLM wrappers return object-like responses.
        content = getattr(response, "content", str(response))
        return _clean_answer_text(content)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        if _is_quota_error(exc):
            try:
                retriever = get_retriever()
                docs = _retrieve_documents(query, retriever)
                return _retrieval_only_fallback_answer(docs, query=query)
            except Exception:
                return (
                    "OpenAI quota exceeded. Please check billing/credits in your OpenAI account, "
                    "then retry."
                )

        return f"Unable to process your question right now: {exc}"