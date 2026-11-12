import asyncio
from io import BytesIO

from fastapi import HTTPException, UploadFile

from app.api.document import read_upload


def test_read_upload_reads_content_in_chunks():
    upload = UploadFile(filename="test.pdf", file=BytesIO(b"content"))

    assert asyncio.run(read_upload(upload)) == b"content"


def test_read_upload_rejects_empty_content():
    upload = UploadFile(filename="empty.pdf", file=BytesIO())

    try:
        asyncio.run(read_upload(upload))
    except HTTPException as error:
        assert error.status_code == 400
        assert error.detail == "Uploaded file is empty."
    else:
        raise AssertionError("Expected an HTTPException")


def test_read_upload_rejects_content_over_limit(monkeypatch):
    monkeypatch.setattr("app.api.document.MAX_PDF_SIZE_BYTES", 5)
    upload = UploadFile(filename="large.pdf", file=BytesIO(b"123456"))

    try:
        asyncio.run(read_upload(upload))
    except HTTPException as error:
        assert error.status_code == 413
        assert error.detail == "File exceeds the maximum upload size."
    else:
        raise AssertionError("Expected an HTTPException")
