"""Extraction du texte et des métadonnées des documents PDF, DOCX et TXT."""

from __future__ import annotations

import io

MAX_PDF_PAGES = 50


def extract_document(data: bytes, extension: str) -> tuple[str, str]:
    """Renvoie (texte, texte des métadonnées utiles à la détection)."""
    if extension == "txt":
        return data.decode("utf-8", errors="replace"), ""
    if extension == "pdf":
        return _extract_pdf(data)
    if extension == "docx":
        return _extract_docx(data)
    raise ValueError(f"Document non pris en charge : .{extension}")


def _extract_pdf(data: bytes) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
    info = reader.metadata or {}
    metadata = "\n".join(str(info.get(key, "")) for key in ("/Producer", "/Creator", "/Author", "/Title"))
    return "\n".join(pages).strip(), metadata


def _extract_docx(data: bytes) -> tuple[str, str]:
    from docx import Document

    document = Document(io.BytesIO(data))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    props = document.core_properties
    metadata = "\n".join(
        str(value or "") for value in (props.author, props.last_modified_by, props.comments, props.title)
    )
    return text.strip(), metadata
