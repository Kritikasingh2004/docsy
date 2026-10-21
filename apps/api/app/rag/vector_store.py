import chromadb

from app.core.config import get_settings

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
