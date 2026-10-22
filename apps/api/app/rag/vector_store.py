import chromadb

from app.core.config import get_settings
from app.rag.schema import DocumentChunk

settings = get_settings()

client = chromadb.PersistentClient(
    path=str(settings.chroma_path),
)

collection = client.get_or_create_collection(
    name="document_chunks",
    configuration={
        "hnsw": {
            "space": "cosine",
        }
    },
    embedding_function=None,
)


def store_chunks(
    *,
    document_id: str,
    filename: str,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> None:
    """Persist chunks, embeddings, and metadata in ChromaDB."""

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count must match embeddings count.")

    if not chunks:
        return

    collection.add(
        ids=[
            f"{document_id}p{chunk.metadata.page_number}:c{chunk.metadata.chunk_index}"
            for chunk in chunks
        ],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "filename": filename,
                "page_number": chunk.metadata.page_number,
                "chunk_index": chunk.metadata.chunk_index,
            }
            for chunk in chunks
        ],
    )
