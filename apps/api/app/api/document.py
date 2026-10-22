from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.ingestion import ingest_pdf
from app.rag.limit import MAX_PDF_SIZE_BYTES

router = APIRouter(prefix="/documents", tags=["Documents"])


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload without exceeding the configured size limits."""
    content = bytearray()

    while True:
        remaining = MAX_PDF_SIZE_BYTES - len(content)
        chunk = await file.read(min(64 * 1024, remaining + 1))

        if not chunk:
            break

        if len(chunk) > remaining:
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
        # Check file extension
        filename = Path(file.filename or "").name

        if Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=415,
                detail="Only PDF files are supported.",
            )

        # Safely read and enforce size limit
        pdf_bytes = await read_upload(file)

        # Validate PDF
        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. The file is not a valid PDF.",
            )

        # Extract chunks, generate embeddings, and store them
        result = ingest_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
        )

        # Return ingestion result
        return {
            "document_id": result.document_id,
            "filename": filename,
            "total_chunks": len(result.chunks),
            "chunks": [
                {
                    "text": chunk.text,
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                }
                for chunk in result.chunks
            ],
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
