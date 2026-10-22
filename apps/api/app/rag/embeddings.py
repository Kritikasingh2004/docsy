import ollama

from app.core.config import ollama_embedding_model
from app.rag.schema import DocumentChunk


def embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Convert document chunks into nomic embeddibg vectors."""

    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    response = ollama.embed(
        model=ollama_embedding_model,
        input=texts,
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    return embeddings
