from io import BytesIO
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from app.main import app
from app.rag.schema import ChunkMetadata, DocumentChunk, IngestionResult

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
    result = IngestionResult(
        document_id="document-123",
        chunks=[
            DocumentChunk(
                text="Sample extracted text from PDF",
                metadata=ChunkMetadata(page_number=1, chunk_index=0),
            )
        ],
    )

    with patch("app.api.document.ingest_pdf", return_value=result) as ingest:
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    ingest.assert_called_once_with(pdf_bytes=pdf_bytes, filename="test.pdf")
    assert data["document_id"] == "document-123"
    assert data["filename"] == "test.pdf"
    assert data["total_chunks"] == 1
    assert data["chunks"] == [
        {
            "text": "Sample extracted text from PDF",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]


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
    with patch("app.api.document.MAX_PDF_SIZE_BYTES", 5):
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


def test_upload_maps_ingestion_value_error_to_unprocessable_entity():
    """Test that ingestion validation errors are returned as a 422 response."""
    pdf_bytes = make_test_pdf()

    with patch(
        "app.api.document.ingest_pdf",
        side_effect=ValueError("No extractable text found."),
    ):
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "No extractable text found."


def test_upload_accepts_case_insensitive_pdf_extension_and_strips_path():
    """Test that only the safe basename and a case-insensitive extension are used."""
    pdf_bytes = make_test_pdf()
    result = IngestionResult(document_id="document-123", chunks=[])

    with patch("app.api.document.ingest_pdf", return_value=result) as ingest:
        response = client.post(
            "/documents/upload",
            files={"file": ("C:\\tmp\\report.PDF", BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "report.PDF"
    ingest.assert_called_once_with(pdf_bytes=pdf_bytes, filename="report.PDF")
