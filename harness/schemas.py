"""Typed I/O contracts for every pipeline stage.

The harness passes these models between stages instead of raw dicts/strings so that
(a) each stage's inputs/outputs are validated, and (b) latency + guardrail telemetry
has a stable shape to log and later aggregate for the P50/P70/P100 analytics.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Stage(str, Enum):
    STT = "stt"
    INPUT_GUARD = "input_guard"
    EMBED_QUERY = "embed_query"
    RETRIEVE = "retrieve"
    GROUNDING_PRECHECK = "grounding_precheck"
    GENERATE = "generate"
    GROUNDING_POSTCHECK = "grounding_postcheck"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    language: str
    query_type: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    is_selected: Optional[bool] = None
    origin_query_id: Optional[str] = None


class STTResult(BaseModel):
    transcript: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    provider: str = "sarvam"
    latency_ms: float = 0.0
    used_fallback: bool = False


class GuardrailResult(BaseModel):
    stage: Stage
    passed: bool
    reason: Optional[str] = None
    details: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None


class RetrievalResult(BaseModel):
    query: str
    strategy_scores: dict[str, float] = Field(default_factory=dict)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    max_dense_score: Optional[float] = None


class GenerationResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    used_fallback: bool = False
    retries: int = 0


class StageTiming(BaseModel):
    stage: Stage
    latency_ms: float


class PipelineResult(BaseModel):
    query_text: Optional[str] = None
    transcript: Optional[str] = None
    answer: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    guardrail_events: list[GuardrailResult] = Field(default_factory=list)
    stage_timings: list[StageTiming] = Field(default_factory=list)
    retrieval_latency_ms: Optional[float] = None
    total_latency_ms: float = 0.0

    def timings_dict(self) -> dict:
        return {t.stage.value: t.latency_ms for t in self.stage_timings}
