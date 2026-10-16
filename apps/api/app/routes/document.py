from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.limit import MAX_PDF_SIZE_BYTES
from app.rag.pdf import process_pdf

router = APIRouter(prefix="/documents", tags=["Documents"])


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload without exceeding the configured size limits."""
    content = bytearray()

    while True:
        reamining = MAX_PDF_SIZE_BYTES - len(content)
        chunk = await file.read(min(64 * 1024, reamining + 1))

        if not chunk:
            break

        if len(chunk) > reamining:
            raise HTTPException(status_code=413, detail="File exceeds the maximum upload size.")

        content.extend(chunk)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return bytes(content)


@router.post("/upload")
async def upload_pdf(
    file: Annotated[UploadFile, File(...)],
):
    try:
        if Path(file.filename or "").suffix.lower() != ".pdf":
            raise HTTPException(status_code=415, detail="Only PDF files are supported.")

        pdf_bytes = await read_upload(file)

        # Reuse our existing PDF processing pipeline

        chunks = process_pdf(pdf_bytes)

        return {
            "filename": Path(file.filename).name,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "text": chunk.text,
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                }
                for chunk in chunks
            ],
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
