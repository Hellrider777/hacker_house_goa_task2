"""Hybrid (dense + BM25) retrieval, fused across chunking strategies with
Reciprocal Rank Fusion, plus a metadata-filtered variant.

This module is the part that has to fit the task's <200ms budget: index loading
happens once at process start (see StrategyIndex.load), so per-query cost is just
query embedding + FAISS search + BM25 scoring + fusion.

Re-ranking choice: a cross-encoder pass would improve precision but typically costs
tens of ms per candidate on CPU, which risks blowing the latency budget at the corpus
sizes used here. We instead use score-based re-ranking (RRF across dense rank, BM25
rank, and strategy) -- documented as a deliberate latency/quality trade-off in the
README, not an oversight.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import bm25s
import faiss
import numpy as np
from bm25s.tokenization import Tokenizer

from harness.schemas import Chunk, RetrievalResult, RetrievedChunk
from retrieval.embed import Embedder

INDEX_DIR = Path(__file__).parent.parent / "indexes"

RRF_K = 60


@dataclass
class StrategyIndex:
    strategy: str
    faiss_index: "faiss.Index"
    bm25: bm25s.BM25
    tokenizer: Tokenizer
    chunks: list[Chunk]

    @classmethod
    def load(cls, lang: str, strategy: str, index_dir: Path) -> "StrategyIndex":
        strat_dir = index_dir / lang / strategy
        faiss_index = faiss.read_index(str(strat_dir / "faiss.index"))

        bm25 = bm25s.BM25.load(str(strat_dir / "bm25s_index"), load_corpus=False, show_progress=False)
        tokenizer = Tokenizer(stopwords=[], stemmer=None)
        tokenizer.load_vocab(str(strat_dir))

        chunks = []
        with (strat_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.model_validate_json(line))
        return cls(strategy=strategy, faiss_index=faiss_index, bm25=bm25, tokenizer=tokenizer, chunks=chunks)


class HybridRetriever:
    """Holds all strategy indexes for one language in memory and answers queries."""

    def __init__(self, language: str, embedder: Embedder, index_dir: Path = INDEX_DIR):
        self.language = language
        self.embedder = embedder
        self.strategies: dict[str, StrategyIndex] = {}
        for strat_dir in sorted((index_dir / language).iterdir()) if (index_dir / language).exists() else []:
            if strat_dir.is_dir():
                self.strategies[strat_dir.name] = StrategyIndex.load(language, strat_dir.name, index_dir)

    def _search_strategy(
        self, strategy: str, query_vec: np.ndarray, query_text: str, top_k: int
    ) -> list[RetrievedChunk]:
        sidx = self.strategies[strategy]
        n = len(sidx.chunks)
        if n == 0:
            return []
        k_dense = min(top_k * 4, n)

        scores, ids = sidx.faiss_index.search(query_vec.reshape(1, -1), k_dense)
        dense_rank = {int(cid): rank for rank, cid in enumerate(ids[0]) if cid != -1}
        # raw cosine similarity (IndexFlatIP over normalized vectors) -- unlike the RRF
        # rank score below, this reflects true relevance magnitude and can be near-zero
        # for genuinely unrelated queries, which the guardrails need to detect off-topic
        # / gibberish input (RRF rank scores are always small-but-nonzero for whatever
        # happens to rank first, regardless of true relevance).
        dense_score = {int(cid): float(sc) for sc, cid in zip(scores[0], ids[0]) if cid != -1}

        query_tokens = sidx.tokenizer.tokenize([query_text], update_vocab=False, show_progress=False)
        bm25_ids, _ = sidx.bm25.retrieve(query_tokens, k=k_dense, show_progress=False)
        bm25_rank = {int(cid): rank for rank, cid in enumerate(bm25_ids[0])}

        candidate_ids = set(dense_rank) | set(bm25_rank)
        fused: list[RetrievedChunk] = []
        for cid in candidate_ids:
            dr = dense_rank.get(cid)
            br = bm25_rank.get(cid)
            rrf = 0.0
            if dr is not None:
                rrf += 1.0 / (RRF_K + dr)
            if br is not None:
                rrf += 1.0 / (RRF_K + br)
            fused.append(
                RetrievedChunk(
                    chunk=sidx.chunks[cid], score=rrf, dense_rank=dr, bm25_rank=br,
                    dense_score=dense_score.get(cid),
                )
            )

        fused.sort(key=lambda rc: rc.score, reverse=True)
        return fused[:top_k]

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        strategies: list[str] | None = None,
        query_type_filter: str | None = None,
        query_vec: np.ndarray | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        strategies = strategies or list(self.strategies.keys())
        strategies = [s for s in strategies if s in self.strategies]

        if query_vec is None:
            query_vec = self.embedder.embed_query(query_text)

        per_strategy_results: dict[str, list[RetrievedChunk]] = {}
        for strat in strategies:
            results = self._search_strategy(strat, query_vec, query_text, top_k=top_k * 2)
            if strat == "metadata_aware" and query_type_filter:
                filtered = [r for r in results if r.chunk.query_type == query_type_filter]
                results = filtered or results  # fall back rather than return nothing
            per_strategy_results[strat] = results

        # fuse across strategies: RRF again, on top of each strategy's already-fused rank
        combined_chunks: dict[str, RetrievedChunk] = {}
        combined_score: dict[str, float] = {}
        strategy_scores: dict[str, float] = {}
        for strat, results in per_strategy_results.items():
            strategy_scores[strat] = results[0].score if results else 0.0
            for rank, rc in enumerate(results):
                key = rc.chunk.chunk_id
                combined_score[key] = combined_score.get(key, 0.0) + 1.0 / (RRF_K + rank)
                combined_chunks.setdefault(key, rc)

        ranked_keys = sorted(combined_score, key=lambda k: combined_score[k], reverse=True)[:top_k]
        final = [
            RetrievedChunk(
                chunk=combined_chunks[k].chunk,
                score=combined_score[k],
                dense_rank=combined_chunks[k].dense_rank,
                bm25_rank=combined_chunks[k].bm25_rank,
                dense_score=combined_chunks[k].dense_score,
            )
            for k in ranked_keys
        ]
        max_dense_score = max(
            (rc.dense_score for rc in final if rc.dense_score is not None), default=None
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query_text,
            strategy_scores=strategy_scores,
            chunks=final,
            latency_ms=latency_ms,
            max_dense_score=max_dense_score,
        )


class RetrieverRegistry:
    """Lazily loads a HybridRetriever per language and keeps it warm in memory."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.embedder = Embedder()
        self._cache: dict[str, HybridRetriever] = {}

    def get(self, language: str) -> HybridRetriever:
        if language not in self._cache:
            self._cache[language] = HybridRetriever(language, self.embedder, self.index_dir)
        return self._cache[language]

    def preload(self, languages: list[str]):
        for lang in languages:
            self.get(lang)
