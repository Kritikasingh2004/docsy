import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        path = str(item.fspath)
        if "tests/unit" in path:
            item.add_marker(pytest.mark.unit)
        elif "tests/integration" in path:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.requires_ollama)
