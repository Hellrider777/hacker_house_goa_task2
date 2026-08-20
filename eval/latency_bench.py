"""Runs a set of test queries through the harness and reports P50/P70/P100 latency,
split into (a) retrieval-stage-only -- the task's <200ms target -- and (b) full
end-to-end (input-guard through generation, text-in). STT latency is reported
separately in stt_latency_bench.py since it depends on network calls to Sarvam and
shouldn't be averaged into the local-pipeline numbers.

Also exercises the guardrail paths: eval/test_queries.json includes off-topic and
unsafe queries so refusal behaviour is part of the same benchmark run, not a separate
untested code path.

Usage:
    python -m eval.latency_bench --languages hi,ta --runs 150
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from harness.build import build_harness

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"
OUT_DIR = Path(__file__).parent / "results"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((p / 100) * (len(s) - 1))))
    return s[idx]


def load_sample_queries(languages: list[str], per_lang: int) -> list[dict]:
    queries = []
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
                queries.append({"text": text, "language": lang, "kind": "in_domain"})
    return queries


def load_adversarial_queries() -> list[dict]:
    if not TEST_QUERIES_PATH.exists():
        return []
    data = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    return data.get("queries", [])


def run_benchmark(languages: list[str], target_runs: int) -> dict:
    harness = build_harness(languages=languages)

    # warm up the local LLM (first call triggers model download/load, which can take
    # minutes) so that one-time cost doesn't pollute the timed percentiles below
    harness.llm._ensure_loaded()

    pool = load_sample_queries(languages, per_lang=max(10, target_runs // max(1, len(languages))))
    pool += load_adversarial_queries()
    if not pool:
        raise SystemExit("No queries available -- run data/prepare_dataset.py first")

    random.shuffle(pool)
    queries = (pool * ((target_runs // max(1, len(pool))) + 1))[:target_runs]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "latency_raw.csv"
    records = []

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "language", "kind", "retrieval_ms", "total_ms", "refused", "refusal_reason"])
        for q in queries:
            result = harness.run_from_text(q["text"], language=q.get("language", languages[0]))
            row = [
                q["text"][:80],
                q.get("language"),
                q.get("kind", "unknown"),
                round(result.retrieval_latency_ms or 0.0, 3),
                round(result.total_latency_ms, 3),
                result.refused,
                result.refusal_reason or "",
            ]
            writer.writerow(row)
            records.append(result)

    retrieval_ms = [r.retrieval_latency_ms for r in records if r.retrieval_latency_ms is not None]
    total_ms = [r.total_latency_ms for r in records]

    summary = {
        "n_runs": len(records),
        "retrieval_only_ms": {
            "p50": _percentile(retrieval_ms, 50),
            "p70": _percentile(retrieval_ms, 70),
            "p100": _percentile(retrieval_ms, 100),
            "under_200ms_target_pct": 100.0 * sum(1 for v in retrieval_ms if v < 200) / len(retrieval_ms) if retrieval_ms else 0.0,
        },
        "end_to_end_ms": {
            "p50": _percentile(total_ms, 50),
            "p70": _percentile(total_ms, 70),
            "p100": _percentile(total_ms, 100),
        },
        "refusal_rate_pct": 100.0 * sum(1 for r in records if r.refused) / len(records),
    }

    (OUT_DIR / "latency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="hi,ta")
    parser.add_argument("--runs", type=int, default=150)
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",")]
    summary = run_benchmark(languages, args.runs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
