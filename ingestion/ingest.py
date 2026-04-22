import os
import sys
from pathlib import Path
from typing import Dict, List

import fitz
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.embeddings import get_embedding_model

try:
    from unstructured.partition.pdf import partition_pdf  # type: ignore[reportMissingImports]
except Exception:
    partition_pdf = None


def _clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _to_page_map(docs: List[Document]) -> Dict[int, Document]:
    page_map: Dict[int, Document] = {}
    for doc in docs:
        try:
            page = int(doc.metadata.get("page", 0))
        except Exception:
            page = 0
        if page > 0:
            page_map[page] = doc
    return page_map


def _char_len(doc: Document) -> int:
    return len((doc.page_content or "").strip())


def _extract_with_unstructured(file_path: str) -> List[Document]:
    if os.getenv("INGEST_SKIP_UNSTRUCTURED", "0") == "1":
        return []

    if partition_pdf is None:
        return []

    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
        )
    except Exception:
        try:
            elements = partition_pdf(filename=file_path, strategy="fast")
        except Exception:
            return []

    pages = {}
    for element in elements:
        text = _clean_text(str(element))
        if not text:
            continue

        page_number = getattr(getattr(element, "metadata", None), "page_number", None) or 1
        pages.setdefault(page_number, []).append(text)

    docs: List[Document] = []
    for page_number, page_texts in sorted(pages.items()):
        page_content = _clean_text("\n".join(page_texts))
        if page_content:
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": file_path,
                        "page": page_number,
                        "extractor": "unstructured",
                    },
                )
            )
    return docs


def _extract_with_pdfplumber(file_path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text_parts: List[str] = []

                layout_text = page.extract_text(layout=True) or ""
                if layout_text.strip():
                    text_parts.append(layout_text)

                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []

                for table in tables:
                    if not table:
                        continue
                    row_lines = []
                    for row in table:
                        cells = [str(cell).strip() if cell is not None else "" for cell in row]
                        if any(cells):
                            row_lines.append(" | ".join(cells))
                    if row_lines:
                        text_parts.append("\n".join(row_lines))

                page_text = _clean_text("\n\n".join(text_parts))
                if page_text:
                    docs.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file_path,
                                "page": idx,
                                "extractor": "pdfplumber",
                            },
                        )
                    )
    except Exception:
        return []
    return docs


def _extract_with_pymupdf(file_path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        with fitz.open(file_path) as pdf_doc:
            for idx in range(len(pdf_doc)):
                page = pdf_doc[idx]
                page_num = idx + 1
                raw_text = page.get_text("text", sort=True)
                if not isinstance(raw_text, str):
                    raw_text = str(raw_text or "")
                page_text = _clean_text(raw_text)
                if page_text:
                    docs.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file_path,
                                "page": page_num,
                                "extractor": "pymupdf",
                            },
                        )
                    )
    except Exception:
        return []
    return docs


def extract_pdf_documents(file_path: str) -> List[Document]:
    unstructured_docs = _extract_with_unstructured(file_path)
    pdfplumber_docs = _extract_with_pdfplumber(file_path)
    pymupdf_docs = _extract_with_pymupdf(file_path)

    if not (unstructured_docs or pdfplumber_docs or pymupdf_docs):
        return []

    unstructured_map = _to_page_map(unstructured_docs)
    pdfplumber_map = _to_page_map(pdfplumber_docs)
    pymupdf_map = _to_page_map(pymupdf_docs)

    all_pages = sorted(set(unstructured_map) | set(pdfplumber_map) | set(pymupdf_map))
    merged_docs: List[Document] = []

    for page in all_pages:
        u_doc = unstructured_map.get(page)
        p_doc = pdfplumber_map.get(page)
        m_doc = pymupdf_map.get(page)

        # Prefer unstructured for RAG-ready segmentation, but replace with richer
        # structured/cleaner extract when text is sparse.
        chosen = u_doc or p_doc or m_doc

        if chosen is None:
            continue

        u_len = _char_len(u_doc) if u_doc else 0
        p_len = _char_len(p_doc) if p_doc else 0
        m_len = _char_len(m_doc) if m_doc else 0

        if u_doc and u_len >= 120:
            chosen = u_doc
        elif p_doc and p_len >= max(120, u_len):
            chosen = p_doc
        elif m_doc and m_len >= max(120, u_len, p_len):
            chosen = m_doc
        elif p_doc and p_len >= u_len:
            chosen = p_doc
        elif m_doc and m_len >= max(u_len, p_len):
            chosen = m_doc

        merged_docs.append(chosen)

    return merged_docs


documents: List[Document] = []
data_path = "data"

for root, _, files in os.walk(data_path):
    for file in files:
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(root, file)
            extracted_docs = extract_pdf_documents(file_path)
            if extracted_docs:
                documents.extend(extracted_docs)
            else:
                print(f"Warning: no text extracted from {file_path}")

if not documents:
    raise ValueError("No PDF text extracted. Check PDF files and extraction dependencies.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embedding = get_embedding_model()
db = FAISS.from_documents(docs, embedding)
db.save_local("vectorstore")