import ollama

from app.core.config import get_settings
from app.rag.schema import DocumentChunk

settings = get_settings()
client = ollama.Client(host=settings.ollama_base_url, timeout=30)


def embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Convert document chunks into embedding vectors."""

    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    response = client.embed(
        model=settings.ollama_embedding_model,
        input=texts,
    )

    embeddings = response["embeddings"]

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    return embeddings
