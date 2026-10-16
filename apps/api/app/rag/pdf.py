import re

import pymupdf

from app.rag.schema import ChunkMetadata, DocumentChunk, DocumentPage


def extract_text_by_page(pdf_bytes: bytes) -> list[DocumentPage]:
    """Extract rext form a PDF while preserving page numbers."""

    pages: list[DocumentPage] = []

    with pymupdf.open(stream=pdf_bytes, filename="pdf") as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            text = re.sub(r"\t+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if text:
                pages.append(
                    DocumentPage(
                        page_number=page_number,
                        text=text,
                    )
                )
    return pages


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[DocumentChunk]:
    """Split pages into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >=0 and < chunk_size.")

    chunks: list[DocumentChunk] = []

    step = chunk_size - overlap

    for page in pages:
        words = page.text.split()

        for chunk_index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                break

            chunks.append(
                DocumentChunk(
                    text=" ".join(chunk_words),
                    metadata=ChunkMetadata(page_number=page.page_number, chunk_index=chunk_index),
                )
            )

            if start + chunk_size > len(words):
                break

    return chunks


def process_pdf(pdf_bytes: bytes) -> list[DocumentChunk]:
    """Extract and chunk a PDF."""

    pages = extract_text_by_page(pdf_bytes)

    if not pages:
        raise ValueError("No extractable text found. The PDF may be scanned or empty.")

    return chunk_pages(pages)
