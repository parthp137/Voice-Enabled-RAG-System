import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

# Global singleton — set during lifespan startup
retrieval_engine: RetrievalEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavyweight resources once at startup, release on shutdown."""
    global retrieval_engine
    logger.info("Starting up: loading RetrievalEngine...")
    retrieval_engine = RetrievalEngine()
    logger.info("RetrievalEngine ready.")
    yield
    retrieval_engine = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="HHG Voice RAG API",
    description="Multilingual Voice Search & RAG Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Import routers after app is created to avoid circular imports
from app.routes_debug import router as debug_router  # noqa: E402

app.include_router(debug_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "hhg-voice-rag",
        "retrieval_loaded": retrieval_engine is not None,
        "strategies": retrieval_engine.available_strategies if retrieval_engine else [],
    }
