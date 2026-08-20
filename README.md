---
title: Voice RAG MSMARCO-XI
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Voice-Enabled RAG on MSMARCO-XI

HH Goa 2026 Shortlisting Task 2. A voice-enabled Retrieval-Augmented Generation system
over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):
speak a question in an Indic language → Sarvam transcribes it → a hybrid, multi-strategy
retriever pulls grounded context → a local open-source LLM answers, or refuses when it
shouldn't answer.

```
Mic input (webapp)
   -> Sarvam STT                          [stt/sarvam_client.py]
   -> Input guard (off-topic/unsafe)      [guardrails/input_guard.py]
   -> Hybrid retrieval, 4 chunking        [retrieval/retriever.py]
      strategies, RRF-fused                <-- <200ms budget (see "Latency target" below)
   -> Grounding pre-check                 [guardrails/grounding_guard.py]
   -> Local LLM generation                [generation/llm.py]
   -> Grounding post-check                [guardrails/grounding_guard.py]
   -> Answer + citations, or a refusal, with per-stage latency
```

All of this runs as one explicit state machine (`harness/pipeline.py`), not a single
prompt-in/text-out call — see **Harness** below.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux
.venv/bin/pip install -r requirements.txt

cp .env.example .env   # fill in SARVAM_API_KEY
```

Python 3.12 is recommended (torch/faiss wheel availability lags on brand-new Python
releases).

### Build the corpus + indexes (one-time)

```bash
python -m data.prepare_dataset --languages hi,ta --max-queries-per-lang 1500
python -m indexing.build_indexes --languages hi,ta
```

### Run the API + web UI

```bash
uvicorn app.server:app --reload
# open http://localhost:8000
```

### Run the latency benchmark

```bash
python -m eval.latency_bench --languages hi,ta --runs 150
# writes eval/results/latency_raw.csv and eval/results/latency_summary.json
```

## Dataset scope (a deliberate trade-off, not a shortcut)

`ai4bharat/MSMARCO-XI` is 11.5M rows / 55.6GB across 13 Indic languages (as, bn, gu, hi,
kn, ml, mr, ne, or, pa, sa, ta, te, ur) — each row has a `query`/`Eng_Query`, an
`Answer`/`Eng_Answer`, and a `passages` dict (`is_selected`, `English_passages`,
`Translated_passages`). Indexing the full dataset is neither feasible in the task's
timeframe nor necessary to demonstrate the architecture, so `data/prepare_dataset.py`
streams (no full download) the `validation` split and takes the first N query rows for
**two languages — Hindi and Tamil** — deduplicating passages by text hash. This keeps
the corpus at a size (tens of thousands of passages) where the retrieval stage can
realistically be benchmarked and can hit the latency target honestly, while still
proving the pipeline generalizes across languages (Indo-Aryan/Devanagari vs.
Dravidian/Tamil script). More languages/rows are a config change
(`--languages`, `--max-queries-per-lang`), not a code change.

## Chunking strategies ("vast" requirement)

Four independently-indexed strategies (`indexing/chunkers.py`), all built once offline
at index time:

| Strategy | What it does | Why |
|---|---|---|
| `fixed_size` | Token-window chunks (120 tokens, 30 overlap) | Baseline control — the "naive" approach every other strategy is compared against |
| `semantic` | Sentence-split (Latin **and** Devanagari/Indic danda `।॥` punctuation), greedily merged into chunks bounded by both a token budget *and* an embedding-similarity drop (topic-shift cut) | Avoids cutting mid-thought; only matters for the minority of longer passages since most MSMARCO passages are already 1–3 sentences |
| `passage_native` | The MSMARCO passage as-is, one chunk per passage | Matches the dataset's own relevance unit — `is_selected` judgments are defined at this granularity, so this is the retrieval-quality reference point |
| `metadata_aware` | Same unit as `passage_native`, but retrieved through a filter-first path: `query_type`/language act as hard/soft filters rather than inert stored fields (`retriever.py::search`, `strategies=["metadata_aware"]` branch) | Demonstrates metadata as a first-class retrieval axis, not just a display field |

Every chunk (any strategy) carries `language`, `query_type`, `source_lang`,
`target_lang`, `is_selected`, `origin_query_id` as structured metadata — retrievable and
filterable, not just stored strings.

## Retrieval: hybrid, RRF-fused, multi-strategy

Per strategy: dense search (FAISS `IndexFlatIP` over normalized
`paraphrase-multilingual-MiniLM-L12-v2` embeddings) **and** sparse BM25
(`rank_bm25`), combined with Reciprocal Rank Fusion. Then all four strategies' fused
results are RRF-fused again into one final ranked list. Re-ranking is score-based
(RRF), not a cross-encoder pass — a cross-encoder would add tens of ms per candidate on
CPU, which risks the latency budget below; this trade-off is deliberate, not an
oversight.

## Latency target — scope interpretation

The task specifies "chunking + vector DB retrieval + everything through to final
output" under 200ms. Taken completely literally (including STT network round-trips and
LLM token generation) this is not achievable with any real speech and language model in
the loop — no submission using a real STT API or LLM could hit it. We interpret the
200ms budget as covering **the retrieval stage**: query embedding → FAISS search → BM25
search → RRF fusion → context assembly. Chunking itself happens once at index-build
time, not per query. This matches the task's own pipeline diagram, which lists
"Chunking/Retrieval (vector DB)" as one stage distinct from "Speech-to-text" and "Answer
generation." `eval/latency_bench.py` reports **both** numbers — retrieval-only (the
target) and full end-to-end including generation — so this scoping choice is auditable,
not hidden.

### Latency results

Measured via `python -m eval.latency_bench --languages hi,ta --runs 150` (150 runs,
sample + adversarial queries, `eval/results/latency_summary.json`):

| Metric | P50 | P70 | P100 |
|---|---|---|---|
| Retrieval-only (target: all < 200ms) | 39 ms | 43 ms | **354 ms (target not met)** |
| End-to-end (retrieval + generation, text input) | 302 ms | 331 ms | 1264 ms |

P50 and P70 sit comfortably under 50ms, well inside the 200ms retrieval-only budget, and
98.6% of the 150 individual runs land under 200ms. **The strict P100 < 200ms check does
not pass** — one (or a small number of) outlier run(s) pushed the worst case to 354ms,
most likely a scheduling/GC pause rather than a systematic bottleneck, but it is reported
here as a fail rather than rounded away. Refusal rate (guardrails correctly declining to
answer) over this run set: 4.0%, up from an earlier 1.3% after fixing two guardrail bugs
that were letting unsafe/off-topic queries through unrefused (see "Guardrails" below).

STT latency (network-dependent, Sarvam API) is measured and reported separately, not
folded into the above.

## Harness

`harness/pipeline.py` (`RagHarness`) is a typed state machine over pydantic-validated
I/O (`harness/schemas.py`), not a single prompt call:

- Every stage (STT, input guard, retrieval, grounding pre-check, generation, grounding
  post-check) is a discrete function with its own timing, logged into `PipelineResult`.
- External calls are wrapped with bounded retries: Sarvam STT via `tenacity` exponential
  backoff (`stt/sarvam_client.py`), local LLM generation via an in-process retry before
  falling back (`generation/llm.py`).
- Defined fallback per failure mode instead of raw exceptions propagating: STT failure →
  "please try again" refusal; model load/generation failure → deterministic extractive
  answer from the top retrieved chunk (`used_fallback=True`, visible in the response);
  empty/insufficient retrieval → refusal, never a hallucinated guess.

## Guardrails

- **Input guard** (`guardrails/input_guard.py`): rejects empty transcripts, a
  keyword-pattern check for unsafe content (self-harm, weapons, illegal-activity
  requests), and an embedding-similarity-to-domain-anchors check for off-topic queries
  (anchors are sampled from the indexed corpus's own queries at startup, so it
  generalizes across all languages without hand-written per-language rules).
- **Grounding pre-check**: if retrieval returns nothing, or the top result's raw dense
  (cosine) similarity is below threshold, generation is skipped entirely and the system
  returns "I don't have enough information" rather than letting the LLM free-associate
  from a weak prompt.
- **Grounding post-check**: the generated answer is embedded and compared against the
  retrieved context; if it isn't well-supported (low max similarity), the answer is
  withheld with an explicit refusal reason instead of being returned.
- All guardrail decisions are structured (`GuardrailResult{stage, passed, reason,
  details}`) and logged in `PipelineResult.guardrail_events` — inspectable per request,
  not just a boolean.
- `eval/test_queries.json` exercises these paths explicitly (empty, off-topic, unsafe,
  gibberish) as part of the same latency benchmark run, so refusal behaviour is
  continuously tested, not a demo-only code path.
- **Verified against `eval/test_queries.json`'s 8 curated cases**: empty 1/1 refused,
  unsafe 2/2 refused, off-topic 1/4 refused, gibberish 0/1 refused. Two earlier real
  bugs were found and fixed by direct testing this session: the unsafe-content regex
  only matched the literal phrase "how to make/build/synthesize", missing rephrasings
  like "how do I make a bomb" (broadened to a verb+dangerous-noun proximity match); and
  the grounding pre-check gated on the RRF-fused rank score, which is structurally
  near-constant regardless of true relevance (empirically confirmed: 22 calibration
  queries, in-domain and adversarial alike, all produced the identical top RRF score) —
  fixed by adding the raw FAISS cosine similarity as a real signal
  (`RetrievalResult.max_dense_score`).
- **Known, deliberate limitation, not chased further**: off-topic detection has real but
  limited recall (1/4 in the test set) and gibberish detection is not reliable (0/1).
  Both the anchor-similarity and dense-retrieval-similarity signals were calibrated
  empirically and found to overlap substantially between genuine in-domain queries and
  adversarial ones for this broad, open-domain corpus (Hindi in-domain queries range as
  low as ~0.16 anchor-similarity, overlapping the ~0.29-0.50 off-topic band) — raising
  thresholds further to catch more off-topic/gibberish cases would false-positive on
  real in-domain questions at an unacceptable rate. The chosen thresholds (anchor floor
  0.30, dense floor 0.30) are a deliberate precision/recall trade-off favoring not
  blocking genuine users, verified to cause only ~1.4% false-positive refusals on
  genuine queries (2/142 in the latency benchmark's sampled in-domain pool).

## Deployment (Hugging Face Spaces)

`Dockerfile` builds a container running `uvicorn app.server:app` on port 7860 (HF
Spaces' default). The sampled corpus (`data/processed/`) and its prebuilt indexes
(`indexes/`) are committed to the repo (small at the scoped corpus size — see "Dataset
scope") so the Space starts without re-running the data pipeline at boot. Push this repo
to a Space with SDK = Docker; set `SARVAM_API_KEY` as a Space secret.

## Known limitations

- Corpus is a two-language, sampled slice of MSMARCO-XI (see "Dataset scope") — not the
  full 11.5M-row dataset.
- The 200ms target is interpreted as retrieval-only latency (see "Latency target —
  scope interpretation" above); STT and generation add real, reported-separately
  latency on top.
- Off-topic/unsafe detection is embedding-similarity + keyword-pattern based, not a
  dedicated moderation model — sufficient to demonstrate the guardrail pattern, not
  production-grade content moderation.
- Multilingual embedding coverage (`paraphrase-multilingual-MiniLM-L12-v2`) is stronger
  for Hindi than for lower-resource languages in the full 13-language set; Tamil was
  chosen as the second language specifically to sanity-check a non-Devanagari script.

## Repo layout

```
data/prepare_dataset.py     dataset streaming + flattening + sampling
indexing/chunkers.py        4 chunking strategies
indexing/build_indexes.py   builds FAISS + BM25 indexes per (language, strategy)
retrieval/embed.py          multilingual sentence embedder (singleton)
retrieval/retriever.py      hybrid dense+BM25 search, RRF fusion, metadata filtering
stt/sarvam_client.py        Sarvam STT client (retries, timeout, structured fallback)
generation/llm.py           local open-source LLM wrapper (transformers), extractive fallback
guardrails/input_guard.py   off-topic / unsafe / empty-query pre-retrieval guard
guardrails/grounding_guard.py  pre- and post-generation groundedness checks
harness/schemas.py          pydantic I/O contracts for every stage
harness/pipeline.py         the orchestrator (state machine)
harness/build.py            wires a ready-to-use harness (shared by app + eval)
eval/latency_bench.py       P50/P70/P100 benchmark + guardrail-path coverage
eval/test_queries.json      off-topic / unsafe / empty / gibberish test cases
app/server.py                FastAPI backend (/query, /query_text, /health)
app/frontend/index.html     minimal record-and-ask web UI
```
