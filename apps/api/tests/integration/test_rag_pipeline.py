import math

import chromadb
import pymupdf
import pytest

from app.core.config import get_settings
from app.rag import vector_store
from app.rag.embeddings import client as ollama_client
from app.rag.embeddings import embed_chunks
from app.rag.ingestion import ingest_pdf
from app.rag.schema import ChunkMetadata, DocumentChunk


def make_pdf(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


@pytest.fixture
def isolated_collection(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="integration_chunks",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    monkeypatch.setattr(vector_store, "collection", collection)
    return collection


def test_real_ollama_embeddings_have_valid_shape_and_values():
    chunks = [
        DocumentChunk(
            text="Docsy indexes local PDF content.",
            metadata=ChunkMetadata(page_number=1, chunk_index=0),
        ),
        DocumentChunk(
            text="ChromaDB stores vectors for retrieval.",
            metadata=ChunkMetadata(page_number=1, chunk_index=1),
        ),
    ]

    embeddings = embed_chunks(chunks)

    assert len(embeddings) == len(chunks)
    assert len(embeddings[0]) > 0
    assert all(len(embedding) == len(embeddings[0]) for embedding in embeddings)
    assert all(math.isfinite(value) for embedding in embeddings for value in embedding)


def test_real_pdf_ingestion_persists_chunks_after_reopening(
    isolated_collection,
    tmp_path,
):
    result = ingest_pdf(
        make_pdf(
            "Docsy stores project requirements on the first page.",
            "The retrieval section describes document filtering.",
        ),
        "requirements.pdf",
    )

    stored = isolated_collection.get(
        where={"document_id": result.document_id},
        include=["documents", "embeddings", "metadatas"],
    )
    assert len(stored["ids"]) == len(result.chunks) == 2
    assert stored["documents"][0].startswith("Docsy stores project requirements")
    assert stored["metadatas"] == [
        {
            "document_id": result.document_id,
            "filename": "requirements.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "document_id": result.document_id,
            "filename": "requirements.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
    ]
    assert len(stored["embeddings"]) == 2
    assert all(len(embedding) > 0 for embedding in stored["embeddings"])
    assert all(
        math.isfinite(float(value)) for embedding in stored["embeddings"] for value in embedding
    )

    reopened = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    reopened_collection = reopened.get_collection("integration_chunks")
    reopened_data = reopened_collection.get(
        where={"document_id": result.document_id},
        include=["documents", "metadatas"],
    )
    assert reopened_data["ids"] == stored["ids"]
    assert reopened_data["documents"] == stored["documents"]
    assert reopened_data["metadatas"] == stored["metadatas"]


def test_real_retrieval_supports_document_filter_and_relevance(
    isolated_collection,
):
    first = [
        DocumentChunk(
            text="The deployment guide explains local PDF indexing.",
            metadata=ChunkMetadata(page_number=1, chunk_index=0),
        )
    ]
    second = [
        DocumentChunk(
            text="The cooking guide explains sourdough fermentation.",
            metadata=ChunkMetadata(page_number=1, chunk_index=0),
        )
    ]
    first_embeddings = embed_chunks(first)
    second_embeddings = embed_chunks(second)
    vector_store.store_chunks(
        document_id="deployment",
        filename="deployment.pdf",
        chunks=first,
        embeddings=first_embeddings,
    )
    vector_store.store_chunks(
        document_id="cooking",
        filename="cooking.pdf",
        chunks=second,
        embeddings=second_embeddings,
    )

    query_embedding = ollama_client.embed(
        model=get_settings().ollama_embedding_model,
        input="How does PDF indexing work?",
    )["embeddings"][0]
    filtered = isolated_collection.query(
        query_embeddings=[query_embedding],
        where={"document_id": "deployment"},
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    assert filtered["metadatas"][0][0]["document_id"] == "deployment"
    assert "PDF indexing" in filtered["documents"][0][0]

    all_documents = isolated_collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    assert all_documents["metadatas"][0][0]["document_id"] == "deployment"
    assert all_documents["distances"][0][0] < all_documents["distances"][0][1]
