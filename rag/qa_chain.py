import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
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


def _rank_documents_for_query(query, docs, max_docs=4, max_chars=1200):
    if not docs:
        return []

    query_text = (query or "").strip().lower()
    keywords = _extract_query_keywords(query)
    scored = []

    for idx, doc in enumerate(docs):
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue

        lowered = text.lower()
        overlap = sum(1 for kw in keywords if kw in lowered)
        exact_phrase_bonus = 2 if query_text and query_text in lowered else 0
        position_bonus = max(0.0, 0.6 - (idx * 0.08))
        score = (overlap * 3.0) + exact_phrase_bonus + position_bonus

        trimmed = text[:max_chars].strip()
        if trimmed:
            scored.append((score, idx, doc, trimmed))

    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected = []
    seen = set()
    for _, _, doc, trimmed in scored:
        dedupe_key = re.sub(r"\s+", " ", trimmed[:220].lower()).strip()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        selected.append((doc, trimmed))
        if len(selected) >= max_docs:
            break

    return selected


def _format_context(query, docs):
    if not docs:
        return "No relevant context was found in the knowledge base."

    ranked_docs = _rank_documents_for_query(query, docs, max_docs=4, max_chars=1200)
    if not ranked_docs:
        return "No relevant context was found in the knowledge base."

    blocks = []
    for i, (doc, chunk) in enumerate(ranked_docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "unknown"))
        source_name = source.replace("\\", "/").split("/")[-1]
        page = metadata.get("page", "?")
        blocks.append(f"[Context {i} | source: {source_name} | page: {page}]\n{chunk}")

    return "\n\n".join(blocks)


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
    """Build a prompt that supports multiple languages for response generation."""
    language_instruction = ""
    
    if response_language.lower() == "kannada":
        language_instruction = (
            "- Answer ONLY in Kannada (ಕನ್ನಡ)\n"
            "- Use clear, simple Kannada language suitable for farmers\n"
        )
    else:
        language_instruction = (
            "- Answer ONLY in English\n"
            "- Use simple, clear language suitable for farmers\n"
        )

    return (
        "You are an agricultural expert assistant.\n\n"
        "Use the context below to answer the farmer's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Instructions:\n"
        f"{language_instruction}"
        "- Provide a clear, practical, and farmer-friendly answer\n"
        "- If context is insufficient, explicitly mention what is missing and then expand using reliable general agricultural knowledge\n"
        "- Do not force a fixed section format unless the user explicitly asks for it\n"
        "- Expand the question intent briefly, then provide a deeper explanation\n"
        "- Write a fuller response in 2-4 connected paragraphs (and use bullets only if truly needed)\n"
        "- Include practical field-level advice such as timing, dosage ranges, and common mistakes where relevant\n\n"
        "Now provide the final answer."
    )


def _force_kannada_farmer_friendly(llm, retrieved_answer):
    """Enforce Kannada-only answer in simple farmer-friendly language."""
    messages = [
        SystemMessage(
            content=(
                "You are an assistant that ALWAYS answers in Kannada. "
                "Do not use English unless explicitly asked. "
                "Use simple, practical, farmer-friendly Kannada."
            )
        ),
        HumanMessage(
            content=(
                "Translate the following answer into Kannada in simple farmer-friendly language:\n\n"
                f"{retrieved_answer}"
            )
        ),
    ]

    translated = llm.invoke(messages)
    translated_text = translated.content if hasattr(translated, "content") else str(translated)
    return _clean_answer_text(translated_text)


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
        context = _format_context(query, docs)
        llm = ChatOpenAI(temperature=0)
        target_language = (response_language or "English").strip()
        intermediate_language = "English" if target_language.lower() == "kannada" else target_language

        primary_response = llm.invoke(
            [
                SystemMessage(content="You are an agricultural expert assistant."),
                HumanMessage(content=_build_prompt(query, context, response_language=intermediate_language)),
            ]
        )

        retrieved_answer = (
            primary_response.content
            if hasattr(primary_response, "content")
            else str(primary_response)
        )
        retrieved_answer = _clean_answer_text(retrieved_answer)

        if target_language.lower() == "kannada":
            return _force_kannada_farmer_friendly(llm, retrieved_answer)

        return retrieved_answer
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

        detail = str(exc).strip() or type(exc).__name__
        if "dimension mismatch" in detail.lower() or "assertionerror" in detail.lower():
            return (
                "Unable to process your question right now: embedding/index mismatch detected. "
                "Please rebuild the vectorstore by running ingestion/ingest.py with your current "
                "EMBEDDING_MODEL setting."
            )

        return f"Unable to process your question right now: {detail}"