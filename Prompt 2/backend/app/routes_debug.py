from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas import TranscriptResult
from app.stt import (
    SarvamAudioFormatError,
    SarvamAuthError,
    SarvamRateLimitError,
    SarvamSTTError,
    SarvamTimeoutError,
    transcribe,
)

router = APIRouter(prefix="/debug", tags=["debug"])


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
