from uuid import uuid4

from app.rag.embeddings import embed_chunks
from app.rag.pdf import process_pdf
from app.rag.schema import IngestionResult
from app.rag.vector_store import store_chunks


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> IngestionResult:
    """Extract, embed, and store a PDF."""

    chunks = process_pdf(pdf_bytes)
    document_id = str(uuid4())
    embeddings = embed_chunks(chunks)

    store_chunks(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
    )

    return IngestionResult(
        document_id=document_id,
        chunks=chunks,
    )
