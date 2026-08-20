"""The orchestrator: wires STT -> input guard -> retrieval -> grounding precheck ->
generation -> grounding postcheck into one explicit state machine with structured,
validated I/O at every step (see schemas.py), per-stage timing, and defined fallback
behaviour on failure at each stage. This is the "harness" required by the task --
not a single prompt-in/text-out call.

Two entry points:
  run_from_audio(audio_bytes, ...)  -- full pipeline including Sarvam STT
  run_from_text(query_text, ...)    -- skips STT, used by latency_bench.py to isolate
                                        the retrieval-stage latency (the <200ms target)
                                        and by any text-input UI path.
"""
from __future__ import annotations

import time

from generation.llm import LocalLLM
from guardrails.grounding_guard import GroundingGuard
from guardrails.input_guard import InputGuard
from harness.schemas import PipelineResult, Stage, StageTiming
from retrieval.retriever import RetrieverRegistry
from stt.sarvam_client import SarvamSTTClient

DEFAULT_LANGUAGE = "hi"


class RagHarness:
    def __init__(
        self,
        retriever_registry: RetrieverRegistry,
        input_guard: InputGuard,
        grounding_guard: GroundingGuard,
        llm: LocalLLM,
        stt_client: SarvamSTTClient | None = None,
        top_k: int = 5,
    ):
        self.retrievers = retriever_registry
        self.input_guard = input_guard
        self.grounding_guard = grounding_guard
        self.llm = llm
        self.stt_client = stt_client or SarvamSTTClient()
        self.top_k = top_k

    def _refuse(self, result: PipelineResult, reason: str) -> PipelineResult:
        result.refused = True
        result.refusal_reason = reason
        result.answer = {
            "empty_query": "I didn't catch a question -- could you try again?",
            "unsafe_content": "I can't help with that request.",
            "off_topic": "That looks outside what this knowledge base covers, so I won't guess.",
            "no_chunks_retrieved": "I couldn't find anything relevant to answer that.",
            "low_relevance_context": "I don't have enough relevant information to answer that confidently.",
            "ungrounded_answer": "I generated an answer but it wasn't well-supported by the retrieved context, so I'm withholding it rather than risk a wrong answer.",
            "stt_failed": "I couldn't transcribe the audio -- please try again.",
        }.get(reason, "I'm not able to answer that.")
        return result

    def run_from_text(self, query_text: str, language: str = DEFAULT_LANGUAGE, query_type: str | None = None) -> PipelineResult:
        result = PipelineResult(query_text=query_text, transcript=query_text)
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        guard = self.input_guard.check(query_text)
        result.guardrail_events.append(guard)
        result.stage_timings.append(StageTiming(stage=Stage.INPUT_GUARD, latency_ms=(time.perf_counter() - t0) * 1000))
        if not guard.passed:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return self._refuse(result, guard.reason or "input_rejected")

        retriever = self.retrievers.get(language)
        t0 = time.perf_counter()
        retrieval = retriever.search(query_text, top_k=self.top_k, query_type_filter=query_type)
        result.stage_timings.append(StageTiming(stage=Stage.RETRIEVE, latency_ms=retrieval.latency_ms))
        result.retrieval_latency_ms = retrieval.latency_ms

        t0 = time.perf_counter()
        pre_guard = self.grounding_guard.check_context_sufficiency(retrieval)
        result.guardrail_events.append(pre_guard)
        result.stage_timings.append(StageTiming(stage=Stage.GROUNDING_PRECHECK, latency_ms=(time.perf_counter() - t0) * 1000))
        if not pre_guard.passed:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return self._refuse(result, pre_guard.reason or "insufficient_context")

        gen = self.llm.generate(query_text, retrieval)
        result.stage_timings.append(StageTiming(stage=Stage.GENERATE, latency_ms=gen.latency_ms))

        t0 = time.perf_counter()
        post_guard = self.grounding_guard.check_answer_grounded(gen.answer, retrieval)
        result.guardrail_events.append(post_guard)
        result.stage_timings.append(StageTiming(stage=Stage.GROUNDING_POSTCHECK, latency_ms=(time.perf_counter() - t0) * 1000))
        if not post_guard.passed and not gen.used_fallback:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return self._refuse(result, post_guard.reason or "ungrounded_answer")

        result.answer = gen.answer
        result.citations = gen.citations
        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return result

    def run_from_audio(self, audio_bytes: bytes, language: str = DEFAULT_LANGUAGE, filename: str = "audio.wav") -> PipelineResult:
        result = PipelineResult()
        t_start = time.perf_counter()

        sarvam_lang = _to_sarvam_language_code(language)
        stt = self.stt_client.transcribe(audio_bytes, filename=filename, language_code=sarvam_lang)
        result.stage_timings.append(StageTiming(stage=Stage.STT, latency_ms=stt.latency_ms))

        if stt.used_fallback or not stt.transcript:
            result.total_latency_ms = (time.perf_counter() - t_start) * 1000
            return self._refuse(result, "stt_failed")

        result.transcript = stt.transcript
        text_result = self.run_from_text(stt.transcript, language=language)
        # merge: keep STT stage timing at the front, extend rest, sum totals
        text_result.transcript = stt.transcript
        text_result.stage_timings = result.stage_timings + text_result.stage_timings
        text_result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return text_result


_LANG_TO_SARVAM = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN",
    "as": "as-IN", "ur": "ur-IN", "ne": "ne-IN",
}


def _to_sarvam_language_code(lang: str) -> str:
    return _LANG_TO_SARVAM.get(lang, "hi-IN")
