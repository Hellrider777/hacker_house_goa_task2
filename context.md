# Session context — Voice RAG on MSMARCO-XI (HH Goa 2026 Task 2)

Read this first when resuming. It captures decisions, current state, and exact next
steps as of this session, so work can continue without re-deriving anything.

## Task (from task_2_hhg.pdf)

Build a voice-enabled RAG system: voice in -> STT -> chunking/retrieval (vector DB) ->
answer generation. Hard requirements: STT via Sarvam or ElevenLabs (pick one); chunking
must be "vast" (multiple strategies, not naive fixed-size); full retrieval pipeline
(chunking+vectorDB+everything to output) under 200ms; report P50/P70/P100 latency over
a reasonable number of test queries; run inside a proper harness (structured
orchestration, retries, structured I/O, error recovery), not a raw prompt call; add
guardrails (off-topic, unsafe input, hallucination/grounding checks) -- system must know
when *not* to answer.

Deadline: **Aug 22, 2026, 11:59 PM**. Submission needs GitHub repo link, live link, 2
videos + Instagram/X promotion with #RAGInGoa -- **the user explicitly said to skip all
of that (videos/promotion/submission form) for now and just build the project.** Don't
work on those unless asked.

Dataset: `ai4bharat/MSMARCO-XI` on HF -- 13 Indic languages, 11.5M rows/55.6GB total.
**Important gotcha already discovered:** there is no per-language HF `config` (only one
"default" builder config via a loading script) -- languages live as separate parquet
files: `hf://datasets/ai4bharat/MSMARCO-XI/{split}/{stem}{val|train}.parquet` (e.g.
`validation/hinval.parquet` for Hindi). `data/prepare_dataset.py` already handles this
via `_LANG_TO_FILE_STEM` + `_hub_parquet_path()`. Don't try `load_dataset(..., lang,
...)` again -- it fails with `BuilderConfig 'hi' not found`.

## Decisions locked in (confirmed with user, don't re-ask)

- STT: **Sarvam AI** (chosen over ElevenLabs for Indic-language coverage)
- Generation: **local open-source LLM**, no external LLM API. Originally planned
  llama.cpp/GGUF but `llama-cpp-python` **failed to build from source on this Windows
  box** (MinGW/gcc 6.3 too old for the vendored ggml C++). Switched to **Hugging Face
  `transformers` + `torch`** (CPU), model `Qwen/Qwen2.5-1.5B-Instruct` (configurable via
  `GEN_MODEL_REPO` env var) -- prebuilt wheels, no native build needed. This is final,
  don't revisit llama.cpp unless the user asks.
- Deployment target: **Hugging Face Spaces** (Docker SDK), not yet actually deployed.
- Languages indexed: **Hindi (hi) + Tamil (ta)** -- 2 of the 13, chosen to cover both
  Devanagari and a non-Devanagari (Dravidian) script. Sample size: 800 queries/lang ->
  ~7950 passages/lang after dedup (see `data/prepare_dataset.py --max-queries-per-lang`,
  currently run with 800; README/plan mentions 1500 as a stretch target if time allows).
- **200ms latency target is interpreted as retrieval-stage-only** (query embed -> FAISS
  -> BM25 -> RRF fusion -> context assembly), NOT including STT or LLM generation. This
  is a documented, deliberate scope call (see README's "Latency target" section) because
  no real STT+LLM pipeline can hit 200ms literally end-to-end. `eval/latency_bench.py`
  reports both retrieval-only and full end-to-end numbers so this is auditable.
- Python **3.12** venv (`.venv/`), not 3.14 (which is installed as the system default but
  has ML-package wheel availability problems). Always invoke via
  `./.venv/Scripts/python.exe`, and prefix commands with `PYTHONIOENCODING=utf-8` on
  Windows/git-bash to avoid `UnicodeEncodeError` crashes when Devanagari/Tamil text hits
  the console.

## What's built (all files exist and are believed correct)

```
data/prepare_dataset.py     streams per-language parquet, flattens passages, dedups, samples
indexing/chunkers.py        4 strategies: fixed_size, semantic, passage_native, metadata_aware
indexing/build_indexes.py   builds FAISS (faiss.IndexFlatIP) + bm25s index per (lang, strategy)
retrieval/embed.py          Embedder wrapping sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
retrieval/retriever.py      HybridRetriever: dense+BM25 RRF fusion per strategy, then RRF across strategies; RetrieverRegistry caches per-language
stt/sarvam_client.py        Sarvam STT client, tenacity retries, structured fallback on failure
generation/llm.py           LocalLLM (transformers), extractive-fallback on load/generation failure
guardrails/input_guard.py   empty/unsafe-keyword/off-topic (embedding-similarity vs domain anchors) check
guardrails/grounding_guard.py  pre-generation context-sufficiency check + post-generation answer-groundedness check
harness/schemas.py          pydantic I/O contracts for every stage (Chunk, STTResult, GuardrailResult, RetrievalResult, GenerationResult, PipelineResult, Stage enum)
harness/pipeline.py         RagHarness: run_from_text / run_from_audio, explicit state machine, per-stage timing, refusal mapping
harness/build.py            build_harness(): wires registry+guards+llm from data/processed/queries_*.jsonl anchors
eval/latency_bench.py       runs sample+adversarial queries N times, computes P50/P70/P100 (retrieval-only AND end-to-end), writes eval/results/
eval/test_queries.json      hand-written off-topic/unsafe/empty/gibberish test cases for guardrail coverage
app/server.py                FastAPI: /query (audio), /query_text, /health, serves app/frontend/
app/frontend/index.html     minimal record-or-type web UI
Dockerfile                   HF Spaces entrypoint (uvicorn on port 7860)
requirements.txt, .env.example, .gitignore, README.md
```

`.gitignore` deliberately does **not** ignore `data/processed/` or `indexes/` -- they're
meant to be committed (small at this sample size) so an HF Space can boot without
re-running the data pipeline. `.venv/` and `.cache/` are ignored.

## Bugs already hit and fixed this session (don't redo)

1. `llama-cpp-python` wouldn't build on Windows (MinGW too old) -> switched to
   `transformers`+`torch`. Also tried the abetlen prebuilt-wheel index -- blocked by
   this sandbox's network (SSL error to github release assets) -- not just a Windows
   problem, don't retry that route either.
2. `load_dataset("ai4bharat/MSMARCO-XI", lang, split=split, streaming=True)` fails --
   no per-language config exists. Fixed via direct parquet path streaming (see above).
3. Retrieval was **too slow** with `rank_bm25` (pure Python): measured P50=297ms,
   P70=344ms, P100=992ms against a ~200ms target, only 15% of queries under target.
   Root cause: `rank_bm25.get_scores()` linear-scans the whole corpus per query, times
   4 strategies. **Fixed** by switching to `bm25s` (scipy-vectorized) -- benchmarked
   standalone at ~10ms for an 8k-doc corpus vs. much worse for rank_bm25. Rewrote
   `indexing/build_indexes.py` and `retrieval/retriever.py` to use
   `bm25s.BM25`/`bm25s.tokenization.Tokenizer` (shared vocab saved/loaded via
   `tokenizer.save_vocab()`/`load_vocab()` so corpus and query tokenization stay
   consistent). **Old `bm25.pkl` files from the rank_bm25 era are stale and unused now
   -- safe to delete** (`find indexes -name bm25.pkl -delete`), retriever.py no longer
   reads them (it reads `bm25s_index/` + `vocab.tokenizer.json` instead).
4. `sentence-transformers` 6.0.0 deprecated `get_sentence_embedding_dimension` in favor
   of `get_embedding_dimension` -- `retrieval/embed.py` handles both via `getattr`
   fallback, already fixed.
5. First `eval/latency_bench.py` run showed `end_to_end_ms.p100` = 263,670ms (~4.4 min)
   -- caused by the first `.generate()` call inside the timed loop triggering the
   one-time Qwen2.5-1.5B model download+load. **Fixed** by adding an explicit
   `harness.llm._ensure_loaded()` warmup call before the timed loop starts in
   `run_benchmark()`. This fix is in the file but **the benchmark has not been re-run
   since this fix** -- do that first when resuming.
6. Windows background-process quirks in this sandbox: `pip install` and dataset-prep
   background bash commands sometimes showed no live output and `tasklist`/`ps` found
   no running process even though the task was genuinely still running -- don't treat
   "no output yet" or "ps shows nothing" as a crash signal on this box; poll via the
   `TaskOutput` tool (`block: true`) instead, which correctly reports `running` vs
   `completed`.

## State as of end of session (exact point to resume from)

Everything achievable without external credentials is now DONE and verified. What's
left is entirely blocked on the user providing something (API key, HF Space, GitHub
auth) -- see the numbered list at the end of this section.

- Dataset prepared, all 8 indexes built (same as before, unchanged this round).
- **Latency benchmark, current numbers** (150 runs, hi+ta, post guardrail-fix --
  superseded pre-fix numbers are in `latency_numbers.txt`'s bottom section for audit
  history, don't use them): retrieval-only P50=39.3ms, P70=43.0ms, P100=354.2ms
  (98.6% of runs under 200ms); end-to-end P50=302.5ms, P70=331.4ms, P100=1264.0ms;
  refusal_rate=4.0%. **retrieval_only_ms.p100 < 200ms still FAILS** (354ms > 200ms) --
  same as before the guardrail fixes, expected, since those fixes don't touch retrieval
  latency. Report this honestly as "target not met on strict P100" with P50/P70/98.6%
  as mitigating context, same framing as before -- don't round this away as "passing."
  README.md's "Latency results" table reflects these current numbers.
- **Guardrail bugs found AND fixed AND re-verified this session** (the important work):
  the task's "must know when not to answer" requirement was found broken during a live
  smoke test, root-caused, fixed, and confirmed improved -- full narrative and exact
  root causes are in `log.txt` (search for "guardrail bug fixes"), don't re-derive it,
  just read that section if detail is needed. Summary of the 3 fixes:
  1. `guardrails/input_guard.py`: unsafe-content regex broadened (was matching only the
     literal phrase "how to make/build/synthesize", missed "how do I make a bomb").
  2. `app/server.py`: `/query_text`'s `text` field changed `Form(...)` -> `Form("")` so
     an empty submission reaches the harness's own graceful empty-query refusal instead
     of raising a raw FastAPI 422 (Starlette's urlencoded form parser was silently
     dropping the empty-valued key before it ever reached guard logic).
  3. `guardrails/grounding_guard.py`'s context-sufficiency check now gates on a real
     signal (`RetrievalResult.max_dense_score`, the raw FAISS cosine similarity) instead
     of the RRF-fused rank score, which was proven structurally useless (identical
     0.0167 value across 22 very different calibration queries). Schema
     (`harness/schemas.py`) and retriever (`retrieval/retriever.py`) were extended to
     capture and propagate this previously-discarded raw score.
  Also raised `input_guard.py`'s anchor-similarity floor from 0.20 to 0.30 (empirically
  calibrated -- see `log.txt` for the full distribution data; genuine Hindi queries
  range as low as ~0.16 similarity, so anything higher than ~0.30 starts
  false-positiving on real queries at an unacceptable rate for this broad corpus).
  **Re-verified** via direct harness testing of `eval/test_queries.json`'s 8 cases:
  empty 1/1, unsafe 2/2, off_topic 1/4, gibberish 0/1 refused correctly (up from
  empty 1/1, unsafe 1/2, off_topic 0/4, gibberish 0/1 before the fixes). Off-topic
  recall (1/4) and gibberish detection (0/1) are **accepted, documented limitations**,
  not further-chaseable bugs -- see README's "Guardrails" section for the honest
  writeup of why (embedding-similarity signals fundamentally overlap between genuine
  and adversarial queries for this broad, open-domain corpus; don't try to "fix" this
  further without a fundamentally different approach, e.g. an LLM-based classifier,
  which would cost real per-query latency the 200ms retrieval budget can't absorb
  anyway -- it would have to sit after the latency-critical path).
  Latency benchmark re-run after these fixes shows refusal_rate rose from 1.33% to
  4.0%, including a ~1.4% false-positive rate on genuine in-domain queries (2/142) from
  the raised anchor floor -- an accepted, deliberate trade-off, documented in README.
- README.md's "Latency results" and "Guardrails" sections both updated with the above
  (current numbers, fix summary, and the honest off-topic/gibberish limitation
  writeup). `log.txt` and `latency_numbers.txt` both updated with the same information
  for audit trail.
- App smoke test: DONE earlier this session (server starts, `/health` and
  `/query_text` work end-to-end with real Hindi text via a Python `urllib` script --
  raw `curl` from Git Bash mangles Devanagari args to `?` characters, a Windows console
  codepage issue, not an app bug; don't test Unicode query text via `curl -F` in this
  Bash tool, use a small Python script with `urllib`/`requests` instead).
- `git init` was run (repo initialized). **Still no commit** -- scratch files need
  cleanup first (see below), then commit. This remains the single biggest concrete
  action left that doesn't need external input.
- Stale `indexes/*/bm25.pkl` files (rank_bm25-era, unused) still present -- harmless,
  low priority, an earlier bulk-delete attempt was blocked by the permission
  classifier.
- **Scratch files in repo root, not yet deleted, must NOT be committed**:
  `scratch_test_query.py`, `scratch_calibrate_grounding.py`,
  `scratch_calibrate_anchor.py`, `scratch_verify_guardrails.py`. `.gitignore` does not
  exclude `scratch_*` yet -- either delete these files or add a gitignore rule before
  the first commit.
- Uvicorn dev server may still be running in the background on port 8000 from earlier
  in the session -- check `curl -s http://localhost:8000/health` before starting a
  duplicate.

- **What's left, in order -- all blocked on the user, nothing left that Claude can do
  unilaterally:**
  1. Clean up scratch files (delete or gitignore) and make the first git commit. This
     one is NOT blocked on the user and should just be done next.
  2. HF Spaces deployment -- needs the user to create a Space and confirm push
     access/credentials.
  3. Real Sarvam voice-path testing -- needs a real `SARVAM_API_KEY`; no `.env` file
     exists yet in the repo root.
  4. GitHub push -- `gh` CLI is not installed/authenticated in this environment
     (`gh: command not found`); needs the user to either install+auth `gh`, push
     manually themselves, or provide a token.

## Commands reference (Git Bash, from project root)

```bash
# check index rebuild status
for d in hi/fixed_size hi/semantic hi/passage_native hi/metadata_aware ta/fixed_size ta/semantic ta/passage_native ta/metadata_aware; do
  [ -d "indexes/$d/bm25s_index" ] && echo "$d DONE" || echo "$d pending"
done

# (re)build indexes if needed -- idempotent, ~20 min total (semantic chunking is the slow strategy, ~8 min/language)
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m indexing.build_indexes --languages hi,ta

# latency benchmark -- THE NEXT THING TO RUN
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m eval.latency_bench --languages hi,ta --runs 150
cat eval/results/latency_summary.json

# run the app
PYTHONIOENCODING=utf-8 ./.venv/Scripts/uvicorn.exe app.server:app --reload
# open http://localhost:8000
```

## Things NOT to re-litigate

- Don't re-ask which STT/LLM/deployment target -- already decided (see above).
- Don't re-attempt llama-cpp-python or the abetlen wheel index.
- Don't try `load_dataset` with a language-code config arg on MSMARCO-XI.
- Don't touch video/promotion/submission-form work unless the user explicitly asks --
  they were explicit that this session is build-only.
- Plan file (from EnterPlanMode/ExitPlanMode) is at
  `C:\Users\krsna\.claude\plans\recursive-imagining-hippo.md` if the original approved
  plan text is needed for reference.
