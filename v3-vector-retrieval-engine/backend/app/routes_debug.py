from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.schemas import RetrievedChunk, TranscriptResult
from app.stt import (
    SarvamAudioFormatError,
    SarvamAuthError,
    SarvamRateLimitError,
    SarvamSTTError,
    SarvamTimeoutError,
    transcribe,
)

router = APIRouter(prefix="/debug", tags=["debug"])


# ---------------------------------------------------------------------------
# STT debug endpoint (Prompt 2)
# ---------------------------------------------------------------------------


@router.post("/transcribe", response_model=TranscriptResult)
async def debug_transcribe(
    file: UploadFile = File(...),
    language_hint: Optional[str] = Form(None),
):
    """Temporary debug route for testing STT in isolation via file upload."""
    if not file.filename:
        filename = "audio.wav"
    else:
        filename = file.filename

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    try:
        result = await transcribe(
            audio_bytes=audio_bytes,
            filename=filename,
            language_hint=language_hint,
        )
        return result
    except SarvamAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    except SarvamAudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except SarvamRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        )
    except SarvamTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        )
    except SarvamSTTError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Retrieval debug endpoint (Prompt 3)
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str
    strategy: str = "ensemble"
    top_k: int = Field(default=5, ge=1, le=50)
    language: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    strategy: str
    results: list[RetrievedChunk]
    count: int


@router.post("/retrieve", response_model=RetrieveResponse)
async def debug_retrieve(req: RetrieveRequest):
    """Debug route for testing vector retrieval in isolation.

    Accepts a text query and returns ranked RetrievedChunk results
    without STT or generation in the loop.
    """
    from app.main import retrieval_engine

    if retrieval_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RetrievalEngine not loaded yet. Wait for startup.",
        )

    try:
        if req.strategy == "ensemble":
            results = retrieval_engine.search_ensemble(
                query=req.query,
                top_k=req.top_k,
                language=req.language,
            )
        else:
            results = retrieval_engine.search(
                query=req.query,
                strategy=req.strategy,
                top_k=req.top_k,
                language=req.language,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return RetrieveResponse(
        query=req.query,
        strategy=req.strategy,
        results=results,
        count=len(results),
    )
