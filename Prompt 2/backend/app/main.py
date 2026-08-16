from fastapi import FastAPI
from app.routes_debug import router as debug_router

app = FastAPI(
    title="HHG Voice RAG API",
    description="Multilingual Voice Search & RAG Service",
    version="1.0.0",
)

app.include_router(debug_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "hhg-voice-rag"}
