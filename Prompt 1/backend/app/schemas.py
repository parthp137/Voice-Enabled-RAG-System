from typing import Any, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class PipelineStageResult(BaseModel):
    stage: str
    ok: bool
    latency_ms: float
    data: Any = None
    error: Optional[str] = None


class AskResponse(BaseModel):
    transcript: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool
    refused: bool
    refusal_reason: Optional[str] = None
    latency_breakdown: dict[str, float] = Field(default_factory=dict)
