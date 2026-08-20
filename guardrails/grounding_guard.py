"""Grounding guardrails: run both before generation (is there enough context to even
try answering?) and after generation (is the answer actually supported by that
context, or did the model drift/hallucinate?).

Both checks are embedding-similarity based (fast, local, no extra API call) rather
than an LLM self-critique call, to keep the harness's error-recovery path cheap and
deterministic.
"""
from __future__ import annotations

import numpy as np

from harness.schemas import GuardrailResult, RetrievalResult, Stage


class GroundingGuard:
    # min_top_score gates on the RRF-fused rank score, which is kept only as a cheap
    # sanity check -- it's structurally near-constant (1/(60+rank)-shaped) regardless of
    # true relevance, confirmed empirically (all of 22 calibration queries, in-domain
    # and adversarial alike, produced the same top RRF score to 4 decimal places), so it
    # can never actually discriminate low-relevance context on its own.
    #
    # min_dense_score gates on the raw FAISS cosine similarity (real relevance signal,
    # unlike the RRF score above). Calibration showed genuine in-domain queries score
    # ~0.52-0.97 and adversarial off-topic queries ~0.43-0.70 -- overlapping ranges, so
    # this is deliberately set low (0.30) as a defense-in-depth check that only catches
    # fully-degenerate near-zero-relevance retrievals, not a primary off-topic detector
    # (that job belongs to InputGuard's anchor-similarity check upstream, itself also
    # limited -- see context.md). Known accepted gap: pure gibberish/OOV text can embed
    # anomalously close to real content in this multilingual model and isn't reliably
    # caught by either signal.
    def __init__(
        self,
        embedder,
        min_top_score: float = 0.01,
        min_dense_score: float = 0.30,
        min_answer_support: float = 0.35,
    ):
        self.embedder = embedder
        self.min_top_score = min_top_score
        self.min_dense_score = min_dense_score
        self.min_answer_support = min_answer_support

    def check_context_sufficiency(self, retrieval: RetrievalResult) -> GuardrailResult:
        if not retrieval.chunks:
            return GuardrailResult(
                stage=Stage.GROUNDING_PRECHECK, passed=False, reason="no_chunks_retrieved"
            )
        top_score = retrieval.chunks[0].score
        details = {"top_score": top_score, "max_dense_score": retrieval.max_dense_score}

        if retrieval.max_dense_score is not None and retrieval.max_dense_score < self.min_dense_score:
            return GuardrailResult(
                stage=Stage.GROUNDING_PRECHECK,
                passed=False,
                reason="low_relevance_context",
                details=details,
            )
        if top_score < self.min_top_score:
            return GuardrailResult(
                stage=Stage.GROUNDING_PRECHECK,
                passed=False,
                reason="low_relevance_context",
                details=details,
            )
        return GuardrailResult(stage=Stage.GROUNDING_PRECHECK, passed=True, details=details)

    def check_answer_grounded(self, answer: str, retrieval: RetrievalResult) -> GuardrailResult:
        if not answer or not answer.strip():
            return GuardrailResult(stage=Stage.GROUNDING_POSTCHECK, passed=False, reason="empty_answer")

        context_texts = [rc.chunk.text for rc in retrieval.chunks]
        if not context_texts:
            return GuardrailResult(stage=Stage.GROUNDING_POSTCHECK, passed=False, reason="no_context")

        answer_vec = self.embedder.embed_query(answer)
        context_vecs = self.embedder.embed_texts(context_texts)
        sims = context_vecs @ answer_vec
        max_support = float(np.max(sims))

        if max_support < self.min_answer_support:
            return GuardrailResult(
                stage=Stage.GROUNDING_POSTCHECK,
                passed=False,
                reason="ungrounded_answer",
                details={"max_support_similarity": max_support},
            )
        return GuardrailResult(
            stage=Stage.GROUNDING_POSTCHECK, passed=True, details={"max_support_similarity": max_support}
        )
