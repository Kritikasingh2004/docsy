import pytest


@pytest.fixture(scope="session", autouse=True)
def require_ollama():
    from app.core.config import get_settings
    from app.rag.embeddings import client

    settings = get_settings()
    try:
        models = client.list().models
    except Exception as exc:
        pytest.skip(
            f"Ollama is unavailable at {settings.ollama_base_url}: "
            f"{type(exc).__name__}: {exc}"
        )

    available = {model.model for model in models}
    if settings.ollama_embedding_model not in available:
        pytest.skip(
            f"Ollama model {settings.ollama_embedding_model!r} is not available at "
            f"{settings.ollama_base_url}. Available models: {sorted(available)}"
        )
