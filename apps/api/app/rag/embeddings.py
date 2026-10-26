import ollama

from app.core.config import get_settings
from app.rag.schema import DocumentChunk

settings = get_settings()

def embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Convert document chunks into nomic embeddibg vectors."""

    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    response = ollama.embed(
        model= settings.ollama_embedding_model,
        input=texts,
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    return embeddings
