"""Builds one FAISS (dense) index + one BM25 (sparse) index per (language, strategy)
from the flattened passage corpus produced by data/prepare_dataset.py.

Indexes are built once, offline; query time only does embed-query + ANN search +
BM25 lookup, which is what has to fit the <200ms retrieval budget. BM25 uses `bm25s`
(scipy-vectorized) rather than `rank_bm25` (pure-Python linear scan) -- the latter
alone pushed retrieval well past 200ms at this corpus size (measured P50 ~297ms across
4 strategies); bm25s cut per-query BM25 lookup to single-digit ms.

Layout written:
    indexes/{lang}/{strategy}/faiss.index
    indexes/{lang}/{strategy}/bm25s_index/   (bm25s.BM25.save() directory)
    indexes/{lang}/{strategy}/vocab.json     (bm25s tokenizer vocab, shared corpus<->query)
    indexes/{lang}/{strategy}/chunks.jsonl   (row i corresponds to faiss id i / bm25 doc i)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bm25s
import faiss
from bm25s.tokenization import Tokenizer
from tqdm import tqdm

from indexing.chunkers import STRATEGIES
from retrieval.embed import Embedder

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
INDEX_DIR = Path(__file__).parent.parent / "indexes"


def load_passages(lang: str) -> list[dict]:
    path = DATA_DIR / f"passages_{lang}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_for_language(lang: str, embedder: Embedder, out_dir: Path):
    rows = load_passages(lang)
    print(f"[{lang}] {len(rows)} source passages")

    for strategy_name, chunk_fn in STRATEGIES.items():
        chunks = []
        for row in tqdm(rows, desc=f"[{lang}/{strategy_name}] chunking"):
            # every strategy function accepts (row, embedder) -- embedder is simply
            # unused by the strategies that don't need it (passage_native, metadata_aware)
            chunks.extend(chunk_fn(row, embedder))

        if not chunks:
            print(f"[{lang}/{strategy_name}] no chunks produced, skipping")
            continue

        texts = [c.text for c in chunks]
        vecs = embedder.embed_texts(texts)

        index = faiss.IndexFlatIP(embedder.dim)
        index.add(vecs)

        # no stopword removal / stemming: corpus mixes English and multiple Indic
        # scripts, so an English-stopword list would silently degrade non-English text
        tokenizer = Tokenizer(stopwords=[], stemmer=None)
        corpus_tokens = tokenizer.tokenize(texts, update_vocab=True, show_progress=False)
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)

        strat_dir = out_dir / lang / strategy_name
        strat_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(strat_dir / "faiss.index"))
        bm25.save(str(strat_dir / "bm25s_index"))
        tokenizer.save_vocab(str(strat_dir))
        with (strat_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")

        print(f"[{lang}/{strategy_name}] indexed {len(chunks)} chunks -> {strat_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="hi,ta")
    parser.add_argument("--out-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    embedder = Embedder()

    for lang in args.languages.split(","):
        build_for_language(lang.strip(), embedder, out_dir)


if __name__ == "__main__":
    main()
