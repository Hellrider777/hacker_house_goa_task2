"""Multiple chunking strategies over the flattened MSMARCO-XI passage table.

Each strategy is a function `(passage_row: dict, embedder) -> list[Chunk]`. All strategies
are run at index-build time (not per-query) and each strategy's chunks are indexed
separately (retrieval/retriever.py fuses across strategies at query time).

Strategies:
  1. fixed_size   - token-window chunking with overlap (baseline).
  2. semantic     - sentence-boundary splitting + greedy merge, cut on embedding
                    similarity drop (topic-shift aware).
  3. passage_native - the MSMARCO passage itself is the chunk (ground-truth granularity
                    for this dataset: passages are already short, curated units).
  4. metadata_aware - wraps passage_native but is indexed with a *restrictive* metadata
                    schema (query_type / source_lang / target_lang / is_selected) used as
                    first-class retrieval filters, not just stored strings.

All strategies attach the same rich metadata to every chunk (language, query_type,
source_lang, target_lang, is_selected, origin_query_id) -- "metadata-aware" is therefore
both a standalone strategy (filter-first retrieval) and a property of every chunk.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from harness.schemas import Chunk

# Sentence boundary across Latin punctuation and the Devanagari/Indic danda (।, ॥)
# used by Hindi, Marathi, Sanskrit, Nepali, etc.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")

# Rough whitespace tokenizer -- good enough for chunk-size budgeting without pulling
# in a per-language tokenizer for 13 Indic languages.
def _word_tokens(text: str) -> list[str]:
    return text.split()


def _make_chunk(doc_id: str, idx: int, text: str, strategy: str, row: dict) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{strategy}::{idx}",
        doc_id=doc_id,
        text=text,
        strategy=strategy,
        language=row["language"],
        query_type=row.get("query_type"),
        source_lang=row.get("source_lang"),
        target_lang=row.get("target_lang"),
        is_selected=row.get("is_selected"),
        origin_query_id=row.get("query_id"),
    )


def fixed_size_chunks(row: dict, embedder=None, chunk_size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Fixed-size token windows with overlap. Baseline strategy required as a control
    against the smarter strategies below."""
    words = _word_tokens(row["text"])
    if not words:
        return []
    if len(words) <= chunk_size:
        return [_make_chunk(row["doc_id"], 0, row["text"], "fixed_size", row)]

    chunks = []
    step = max(1, chunk_size - overlap)
    idx = 0
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(_make_chunk(row["doc_id"], idx, " ".join(window), "fixed_size", row))
        idx += 1
        if start + chunk_size >= len(words):
            break
    return chunks


def semantic_chunks(
    row: dict,
    embedder,
    max_tokens: int = 150,
    similarity_threshold: float = 0.55,
) -> list[Chunk]:
    """Split into sentences, then greedily merge consecutive sentences into a chunk
    until either the token budget is hit or embedding similarity between the running
    chunk and the next sentence drops below `similarity_threshold` (topic shift).

    Falls back to passage_native behaviour for short passages (most MSMARCO passages
    are 1-3 sentences already, so semantic splitting mainly matters for the small
    fraction of long/concatenated passages).
    """
    text = row["text"]
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return [_make_chunk(row["doc_id"], 0, text, "semantic", row)]

    embeddings = embedder.embed_texts(sentences)
    chunks: list[Chunk] = []
    idx = 0
    current_sents = [sentences[0]]
    current_vec = embeddings[0]
    current_tokens = len(_word_tokens(sentences[0]))

    for i in range(1, len(sentences)):
        sim = float(np.dot(current_vec, embeddings[i]) / (
            np.linalg.norm(current_vec) * np.linalg.norm(embeddings[i]) + 1e-8
        ))
        sent_tokens = len(_word_tokens(sentences[i]))
        fits_budget = current_tokens + sent_tokens <= max_tokens
        same_topic = sim >= similarity_threshold

        if fits_budget and same_topic:
            current_sents.append(sentences[i])
            current_tokens += sent_tokens
            # running centroid, not just last sentence, so drift is smoothed
            n = len(current_sents)
            current_vec = (current_vec * (n - 1) + embeddings[i]) / n
        else:
            chunks.append(_make_chunk(row["doc_id"], idx, " ".join(current_sents), "semantic", row))
            idx += 1
            current_sents = [sentences[i]]
            current_vec = embeddings[i]
            current_tokens = sent_tokens

    if current_sents:
        chunks.append(_make_chunk(row["doc_id"], idx, " ".join(current_sents), "semantic", row))
    return chunks


def passage_native_chunks(row: dict, embedder=None) -> list[Chunk]:
    """The MSMARCO passage as-is: one chunk per source passage. This is the closest
    match to how the dataset's own relevance judgments (`is_selected`) are defined,
    so it acts as the retrieval-quality baseline against the other strategies."""
    return [_make_chunk(row["doc_id"], 0, row["text"], "passage_native", row)]


def metadata_aware_chunks(row: dict, embedder=None) -> list[Chunk]:
    """Same text unit as passage_native, but indexed as a distinct strategy whose
    retriever variant (retrieval/retriever.py::search) applies metadata filters
    (language / query_type / source_lang) *before* vector search rather than treating
    metadata as inert stored fields. Kept as its own strategy so filtered vs.
    unfiltered retrieval can be compared directly in eval."""
    return [_make_chunk(row["doc_id"], 0, row["text"], "metadata_aware", row)]


STRATEGIES: dict[str, Callable] = {
    "fixed_size": fixed_size_chunks,
    "semantic": semantic_chunks,
    "passage_native": passage_native_chunks,
    "metadata_aware": metadata_aware_chunks,
}
