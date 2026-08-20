"""Wires up a ready-to-use RagHarness: loads retriever indexes, builds the input guard's
domain anchors from the indexed corpus's own queries, and constructs the local LLM.
Shared by the FastAPI app and the latency benchmark so both exercise the exact same
harness configuration.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from generation.llm import LocalLLM
from guardrails.grounding_guard import GroundingGuard
from guardrails.input_guard import InputGuard
from harness.pipeline import RagHarness
from retrieval.retriever import RetrieverRegistry

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def _load_domain_anchors(languages: list[str], per_lang: int = 60) -> list[str]:
    anchors = []
    for lang in languages:
        path = DATA_DIR / f"queries_{lang}.jsonl"
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        sample = random.sample(lines, min(per_lang, len(lines))) if lines else []
        for line in sample:
            row = json.loads(line)
            text = row.get("query_text") or row.get("eng_query")
            if text:
                anchors.append(text)
    return anchors


def build_harness(languages: list[str] | None = None, top_k: int = 5) -> RagHarness:
    languages = languages or ["hi", "ta"]
    registry = RetrieverRegistry()
    registry.preload(languages)

    anchors = _load_domain_anchors(languages)
    input_guard = InputGuard(registry.embedder, anchors)
    grounding_guard = GroundingGuard(registry.embedder)
    llm = LocalLLM()

    return RagHarness(
        retriever_registry=registry,
        input_guard=input_guard,
        grounding_guard=grounding_guard,
        llm=llm,
        top_k=top_k,
    )
