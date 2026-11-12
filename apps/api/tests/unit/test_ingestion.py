from unittest.mock import patch

from app.rag.ingestion import ingest_pdf
from app.rag.schema import ChunkMetadata, DocumentChunk


def test_ingest_pdf_processes_embeds_and_stores_chunks():
    chunks = [
        DocumentChunk(
            text="extracted text",
            metadata=ChunkMetadata(page_number=1, chunk_index=0),
        )
    ]
    embeddings = [[0.1, 0.2]]

    with (
        patch("app.rag.ingestion.process_pdf", return_value=chunks) as process,
        patch("app.rag.ingestion.embed_chunks", return_value=embeddings) as embed,
        patch("app.rag.ingestion.store_chunks") as store,
        patch("app.rag.ingestion.uuid4", return_value="document-123"),
    ):
        result = ingest_pdf(b"%PDF-test", "test.pdf")

    assert result.document_id == "document-123"
    assert result.chunks == chunks
    process.assert_called_once_with(b"%PDF-test")
    embed.assert_called_once_with(chunks)
    store.assert_called_once_with(
        document_id="document-123",
        filename="test.pdf",
        chunks=chunks,
        embeddings=embeddings,
    )
