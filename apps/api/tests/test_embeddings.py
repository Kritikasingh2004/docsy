from unittest.mock import patch

from app.rag.embeddings import embed_chunks
from app.rag.schema import ChunkMetadata, DocumentChunk


def make_chunk(text: str = "chunk") -> DocumentChunk:
    return DocumentChunk(
        text=text,
        metadata=ChunkMetadata(page_number=1, chunk_index=0),
    )


def test_embed_chunks_returns_empty_list_without_calling_ollama():
    with patch("app.rag.embeddings.ollama.embed") as embed:
        assert embed_chunks([]) == []

    embed.assert_not_called()


def test_embed_chunks_embeds_chunk_texts():
    chunks = [make_chunk("first"), make_chunk("second")]

    with (
        patch("app.rag.embeddings.settings.ollama_embedding_model", "test-model"),
        patch(
            "app.rag.embeddings.ollama.embed",
            return_value={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        ) as embed,
    ):
        result = embed_chunks(chunks)

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    embed.assert_called_once_with(
        model="test-model",
        input=["first", "second"],
    )


def test_embed_chunks_rejects_mismatched_embedding_count():
    with patch(
        "app.rag.embeddings.ollama.embed",
        return_value={"embeddings": [[0.1, 0.2]]},
    ):
        try:
            embed_chunks([make_chunk("first"), make_chunk("second")])
        except RuntimeError as error:
            assert str(error) == "Embedding count does not match chunk count."
        else:
            raise AssertionError("Expected a RuntimeError")
