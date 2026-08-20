"""Pulls a manageable, deduplicated slice of ai4bharat/MSMARCO-XI and flattens it into
a per-language passage corpus + query set.

The full dataset is 11.5M rows / 55.6GB across 13 languages -- far more than is useful
or feasible to index for a demo in the task's timeframe. The Hub repo has no per-language
*config* (only a single "default" builder config backed by a loading script) -- instead
each language is a separate parquet file under validation/<code>val.parquet. We stream
that file directly via the `hf://` fsspec path (no full-file download) and take the
first N query rows, then explode each row's `passages` (English_passages /
Translated_passages / is_selected) into one passage record per list entry. This scope
decision is documented in the README.

Usage:
    python -m data.prepare_dataset --languages hi,ta --max-queries-per-lang 1500
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

OUT_DIR = Path(__file__).parent / "processed"

# ai4bharat language code -> parquet filename stem on the Hub repo
# (validation/<stem>val.parquet, train/<stem>train.parquet)
_LANG_TO_FILE_STEM = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def _hub_parquet_path(lang: str, split: str) -> str:
    if lang not in _LANG_TO_FILE_STEM:
        raise ValueError(f"Unknown language code '{lang}', expected one of {sorted(_LANG_TO_FILE_STEM)}")
    stem = _LANG_TO_FILE_STEM[lang]
    suffix = "val" if split == "validation" else "train"
    return f"hf://datasets/ai4bharat/MSMARCO-XI/{split}/{stem}{suffix}.parquet"


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return bool(v)


def flatten_language(lang: str, split: str, max_queries: int, out_dir: Path) -> dict:
    from datasets import load_dataset

    path = _hub_parquet_path(lang, split)
    ds = load_dataset("parquet", data_files={split: path}, split=split, streaming=True)

    passages_path = out_dir / f"passages_{lang}.jsonl"
    queries_path = out_dir / f"queries_{lang}.jsonl"
    seen_hashes: set[str] = set()

    n_queries = 0
    n_passages = 0
    with passages_path.open("w", encoding="utf-8") as pf, queries_path.open(
        "w", encoding="utf-8"
    ) as qf:
        for row in tqdm(ds, total=max_queries, desc=f"[{lang}] streaming"):
            if n_queries >= max_queries:
                break
            passages = row.get("passages") or {}
            translated = passages.get("Translated_passages") or []
            english = passages.get("English_passages") or []
            is_selected = passages.get("is_selected") or []

            texts = translated if any(t and t.strip() for t in translated) else english
            if not texts:
                continue

            query_id = str(row.get("query_id"))
            selected_doc_ids = []
            kept_any = False
            for idx, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                h = _text_hash(text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                selected = _as_bool(is_selected[idx]) if idx < len(is_selected) else False
                doc_id = f"{lang}_{query_id}_{idx}"
                record = {
                    "doc_id": doc_id,
                    "language": lang,
                    "query_id": query_id,
                    "query_type": row.get("query_type"),
                    "source_lang": row.get("source_lang"),
                    "target_lang": row.get("target_lang"),
                    "is_selected": selected,
                    "text": text.strip(),
                }
                pf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_passages += 1
                kept_any = True
                if selected:
                    selected_doc_ids.append(doc_id)

            if not kept_any:
                continue

            query_record = {
                "query_id": query_id,
                "language": lang,
                "query_type": row.get("query_type"),
                "query_text": row.get("query"),
                "eng_query": row.get("Eng_Query"),
                "answer": row.get("Answer"),
                "eng_answer": row.get("Eng_Answer"),
                "selected_doc_ids": selected_doc_ids,
            }
            qf.write(json.dumps(query_record, ensure_ascii=False) + "\n")
            n_queries += 1

    return {"language": lang, "queries": n_queries, "passages": n_passages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="hi,ta", help="comma-separated ai4bharat language codes")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-queries-per-lang", type=int, default=1500)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for lang in args.languages.split(","):
        lang = lang.strip()
        stats = flatten_language(lang, args.split, args.max_queries_per_lang, out_dir)
        summary.append(stats)
        print(f"[{lang}] queries={stats['queries']} passages={stats['passages']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
