from unittest.mock import patch

import pytest

from app.rag.schema import ChunkMetadata, DocumentChunk
from app.rag.vector_store import store_chunks


def make_chunk(text: str = "chunk", chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        metadata=ChunkMetadata(page_number=1, chunk_index=chunk_index),
    )


def test_store_chunks_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="Chunk count must match embeddings count"):
        store_chunks(
            document_id="document-123",
            filename="test.pdf",
            chunks=[make_chunk()],
            embeddings=[],
        )


def test_store_chunks_does_not_write_empty_chunks():
    with patch("app.rag.vector_store.collection.add") as add:
        store_chunks(
            document_id="document-123",
            filename="test.pdf",
            chunks=[],
            embeddings=[],
        )

    add.assert_not_called()


def test_store_chunks_writes_documents_embeddings_and_metadata():
    chunks = [make_chunk("first"), make_chunk("second", chunk_index=1)]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    with patch("app.rag.vector_store.collection.add") as add:
        store_chunks(
            document_id="document-123",
            filename="test.pdf",
            chunks=chunks,
            embeddings=embeddings,
        )

    add.assert_called_once_with(
        ids=["document-123p1:c0", "document-123p1:c1"],
        documents=["first", "second"],
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": "document-123",
                "filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
            {
                "document_id": "document-123",
                "filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        ],
    )
