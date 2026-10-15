from fastapi import FastAPI

app = FastAPI(
    title="Docsy API",
    description="Local-first AI document assistant",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
