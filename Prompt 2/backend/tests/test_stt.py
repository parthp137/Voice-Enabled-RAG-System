import os
import sys
from pathlib import Path
from unittest.mock import patch
import httpx
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas import TranscriptResult
from app.stt import (
    SarvamAudioFormatError,
    SarvamAuthError,
    SarvamRateLimitError,
    SarvamSTTError,
    SarvamTimeoutError,
    transcribe,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# UNIT TESTS (Mocked Sarvam API Responses)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_api_key(monkeypatch):
    """Assert SarvamAuthError is raised when SARVAM_API_KEY is missing."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(SarvamAuthError, match="SARVAM_API_KEY environment variable is not set"):
        await transcribe(b"dummy audio data", "test.wav")


@pytest.mark.asyncio
async def test_http_401_auth_failure(monkeypatch):
    """Assert HTTP 401 returns SarvamAuthError without retrying."""
    monkeypatch.setenv("SARVAM_API_KEY", "invalid_key")
    mock_response = httpx.Response(401, text="Unauthorized: Invalid API Key")

    with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
        with pytest.raises(SarvamAuthError, match="authorization failed"):
            await transcribe(b"dummy audio data", "test.wav")


@pytest.mark.asyncio
async def test_http_400_audio_format_error(monkeypatch):
    """Assert HTTP 400 with audio format error returns SarvamAudioFormatError."""
    monkeypatch.setenv("SARVAM_API_KEY", "test_key")
    mock_response = httpx.Response(400, text="Invalid audio format or corrupt header")

    with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
        with pytest.raises(SarvamAudioFormatError, match="Invalid audio format"):
            await transcribe(b"dummy audio data", "bad.wav")


@pytest.mark.asyncio
async def test_successful_mocked_transcription(monkeypatch):
    """Assert valid 200 response parses transcript and latency correctly."""
    monkeypatch.setenv("SARVAM_API_KEY", "test_key")
    mock_response = httpx.Response(
        200,
        json={
            "transcript": "नमस्ते यह एक परीक्षण है",
            "language_code": "hi-IN",
        },
    )

    with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
        result = await transcribe(b"dummy audio data", "hin_query.wav", language_hint="hi")
        assert isinstance(result, TranscriptResult)
        assert result.text == "नमस्ते यह एक परीक्षण है"
        assert result.detected_language == "hi-IN"
        assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_retry_on_timeout(monkeypatch):
    """Assert SarvamTimeoutError triggers tenacity retry up to max attempts."""
    monkeypatch.setenv("SARVAM_API_KEY", "test_key")
    
    with patch.object(httpx.AsyncClient, "post", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(SarvamTimeoutError, match="timed out"):
            await transcribe(b"dummy audio data", "test.wav")


# ---------------------------------------------------------------------------
# INTEGRATION TESTS (Hits Real Sarvam API with Audio Fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture_name, language_hint",
    [
        ("eng_query.wav", "en"),
        ("hin_query.wav", "hi"),
        ("noisy_mumbled.wav", None),
        ("weather_offtopic.wav", "en"),
        ("short_sample.wav", None),
    ],
)
async def test_live_sarvam_stt_fixtures(fixture_name, language_hint):
    """Integration test hitting Sarvam STT API with real test fixtures."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("Skipping live STT integration test: SARVAM_API_KEY not found in environment.")

    fixture_path = FIXTURES_DIR / fixture_name
    assert fixture_path.exists(), f"Test fixture {fixture_name} missing from {FIXTURES_DIR}"

    audio_bytes = fixture_path.read_bytes()
    assert len(audio_bytes) > 0

    result = await transcribe(
        audio_bytes=audio_bytes,
        filename=fixture_name,
        language_hint=language_hint,
    )

    assert isinstance(result, TranscriptResult)
    assert isinstance(result.text, str)
    assert result.latency_ms > 0

    print(f"\n[STT Integration] Fixture: {fixture_name} | Latency: {result.latency_ms:.2f}ms | Transcript: '{result.text}'")
