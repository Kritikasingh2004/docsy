from io import BytesIO
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_test_pdf(text: str = "Sample extracted text from PDF") -> bytes:
    """Create a real, minimal valid PDF in-memory for testing."""
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# -------------------------
# Upload Endpoint Tests
# -------------------------


def test_upload_success():
    """Test successful upload and processing of a valid PDF."""
    pdf_bytes = make_test_pdf("Sample extracted text from PDF")

    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["total_chunks"] >= 1
    assert any("Sample extracted text" in chunk["text"] for chunk in data["chunks"])


def test_upload_wrong_extension():
    """Test that non-PDF extensions are rejected with 415."""
    response = client.post(
        "/documents/upload",
        files={"file": ("malware.txt", BytesIO(b"some text"), "text/plain")},
    )

    assert response.status_code == 415
    assert "Only PDF files are supported" in response.json()["detail"]


def test_upload_invalid_magic_bytes():
    """Test that a file with a .pdf extension but fake content is rejected with 400."""
    fake_pdf_content = b"This is just a text file disguised as a pdf"

    response = client.post(
        "/documents/upload",
        files={"file": ("fake.pdf", BytesIO(fake_pdf_content), "application/pdf")},
    )

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_upload_exceeds_size_limit():
    """Test that files exceeding the size limit trigger a 413 error."""
    with patch("app.routes.document.MAX_PDF_SIZE_BYTES", 5):
        large_content = b"%PDF-1234567890"  # 15 bytes, which is > 5 bytes limit

        response = client.post(
            "/documents/upload",
            files={"file": ("large.pdf", BytesIO(large_content), "application/pdf")},
        )

        assert response.status_code == 413
        assert "exceeds the maximum upload size" in response.json()["detail"]


def test_upload_empty_file():
    """Test that uploading an empty file gets caught by validation."""
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 400
