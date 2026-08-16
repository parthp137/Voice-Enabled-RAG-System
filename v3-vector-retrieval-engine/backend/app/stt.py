import os
import time
import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.schemas import TranscriptResult

load_dotenv()

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTTError(Exception):
    """Base exception for Sarvam STT service errors."""
    pass


class SarvamAuthError(SarvamSTTError):
    """Raised on HTTP 401 / 403 authorization failures or missing API key."""
    pass


class SarvamAudioFormatError(SarvamSTTError):
    """Raised on invalid audio format or corrupt file (HTTP 400 format errors)."""
    pass


class SarvamTimeoutError(SarvamSTTError):
    """Raised when Sarvam API request times out."""
    pass


class SarvamRateLimitError(SarvamSTTError):
    """Raised on HTTP 429 rate limit error."""
    pass


@retry(
    retry=retry_if_exception_type((SarvamTimeoutError, SarvamRateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
async def _execute_sarvam_request(
    audio_bytes: bytes,
    filename: str,
    language_hint: str | None = None,
) -> tuple[str, str | None, float]:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SarvamAuthError("SARVAM_API_KEY environment variable is not set.")

    headers = {
        "api-subscription-key": api_key
    }

    # Determine MIME type from filename extension
    ext = os.path.splitext(filename)[1].lower()
    mime_type = "audio/wav"
    if ext in [".mp3", ".mpeg"]:
        mime_type = "audio/mpeg"
    elif ext in [".ogg", ".opus"]:
        mime_type = "audio/ogg"
    elif ext in [".m4a", ".aac"]:
        mime_type = "audio/mp4"

    files = {
        "file": (filename, audio_bytes, mime_type)
    }

    data = {
        "model": "saaras:v3",
        "mode": "transcribe"
    }

    if language_hint:
        if language_hint.lower() in ["hi", "hindi", "hi-in"]:
            data["language_code"] = "hi-IN"
        elif language_hint.lower() in ["en", "english", "en-in"]:
            data["language_code"] = "en-IN"
        else:
            data["language_code"] = language_hint

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            response = await client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
    except httpx.TimeoutException as exc:
        raise SarvamTimeoutError(f"Sarvam STT API timed out after 8.0s: {exc}") from exc
    except httpx.RequestError as exc:
        raise SarvamTimeoutError(f"Sarvam STT connection error: {exc}") from exc

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    status_code = response.status_code

    if status_code in (401, 403):
        raise SarvamAuthError(f"Sarvam STT authorization failed (HTTP {status_code}): {response.text}")
    elif status_code == 429:
        raise SarvamRateLimitError(f"Sarvam STT rate limit exceeded (HTTP 429): {response.text}")
    elif status_code == 400:
        msg = response.text.lower()
        if any(term in msg for term in ["audio", "format", "invalid", "corrupt", "decode", "file"]):
            raise SarvamAudioFormatError(f"Invalid audio format submitted (HTTP 400): {response.text}")
        raise SarvamSTTError(f"Sarvam STT bad request (HTTP 400): {response.text}")
    elif status_code != 200:
        raise SarvamSTTError(f"Sarvam STT API error (HTTP {status_code}): {response.text}")

    try:
        resp_json = response.json()
    except Exception as exc:
        raise SarvamSTTError(f"Failed to parse Sarvam STT JSON response: {response.text}") from exc

    transcript_text = resp_json.get("transcript", "").strip()
    detected_language = resp_json.get("language_code") or language_hint

    return transcript_text, detected_language, latency_ms


async def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language_hint: str | None = None,
) -> TranscriptResult:
    text, detected_language, latency_ms = await _execute_sarvam_request(
        audio_bytes=audio_bytes,
        filename=filename,
        language_hint=language_hint,
    )
    return TranscriptResult(
        text=text,
        detected_language=detected_language,
        latency_ms=latency_ms,
    )
