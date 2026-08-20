"""Cheap, local pre-retrieval guardrail: catches empty transcripts, unsafe/inappropriate
input, and off-topic queries before any retrieval or generation cost is spent.

Domain-relevance check is embedding-similarity against a small set of in-domain anchor
queries (sampled from the indexed corpus's own query set at startup) rather than a
hand-written keyword list, so it generalizes across all 13 possible languages/topics in
MSMARCO-XI without per-language rules.
"""
from __future__ import annotations

import re

import numpy as np

from harness.schemas import GuardrailResult, Stage

# Deliberately conservative unsafe-content patterns (self-harm, violence-for-harm,
# explicit illegal activity requests). This is a lightweight keyword layer, not a
# substitute for a full moderation model -- documented as a known limitation.
_UNSAFE_PATTERNS = [
    # verb + dangerous-noun proximity, independent of exact "how to" phrasing
    # (catches "how do I make a bomb", "how can I build a weapon", "make a bomb", etc.)
    r"\b(make|build|create|construct|synthesi[sz]e|manufacture|assemble|acquire|obtain)\b"
    r".{0,30}\b(a |an |the )?(bomb|explosive|poison|chemical weapon|biological weapon|nerve agent|weapon|gun|firearm)\b",
    r"\bkill (myself|yourself|someone|somebody)\b",
    r"\bhurt (myself|yourself)\b",
    r"\b(suicide|self[- ]harm)\b",
    r"\bchild (abuse|exploitation|pornography)\b",
    r"\bhack (into|someone'?s)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


class InputGuard:
    # Calibrated empirically against eval/test_queries.json + sampled in-domain
    # queries: genuine Hindi queries range ~0.16-0.97 similarity to domain anchors
    # (Tamil is more cleanly separated, ~0.58+), while adversarial off-topic English
    # queries range ~0.29-0.50. The two distributions overlap substantially -- there is
    # no threshold that cleanly separates them for this broad, open-domain corpus.
    # 0.30 is a deliberately conservative pick: it recovers some real off-topic
    # detection (previously 0/4 caught at floor=0.20) while keeping the false-positive
    # rate on genuine in-domain queries low (~5% in calibration). See context.md for
    # the calibration data this was picked from.
    def __init__(self, embedder, domain_anchor_texts: list[str], similarity_floor: float = 0.30):
        self.embedder = embedder
        self.similarity_floor = similarity_floor
        self._anchor_vecs = embedder.embed_texts(domain_anchor_texts) if domain_anchor_texts else None

    def check(self, text: str) -> GuardrailResult:
        stripped = (text or "").strip()
        if not stripped:
            return GuardrailResult(stage=Stage.INPUT_GUARD, passed=False, reason="empty_query")

        if _UNSAFE_RE.search(stripped):
            return GuardrailResult(
                stage=Stage.INPUT_GUARD, passed=False, reason="unsafe_content",
                details={"matched_pattern": True},
            )

        if self._anchor_vecs is not None and len(self._anchor_vecs) > 0:
            qvec = self.embedder.embed_query(stripped)
            sims = self._anchor_vecs @ qvec
            max_sim = float(np.max(sims))
            if max_sim < self.similarity_floor:
                return GuardrailResult(
                    stage=Stage.INPUT_GUARD, passed=False, reason="off_topic",
                    details={"max_domain_similarity": max_sim},
                )

        return GuardrailResult(stage=Stage.INPUT_GUARD, passed=True)
