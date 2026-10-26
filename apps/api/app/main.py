from fastapi import FastAPI

from app.api.document import router as document_router

app = FastAPI(
    title="Docsy API",
    description="Local-first AI document assistant",
    version="0.1.0",
)

app.include_router(document_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
