"""Local, open-source generation model (no external LLM API).

Uses Hugging Face `transformers` with a small instruct model on CPU -- chosen over
llama.cpp/GGUF because transformers ships prebuilt PyTorch wheels on every platform
(no native C++ toolchain needed to install), which matters for a fast, reliable setup
under a hard deadline and for HF Spaces' own build environment.

Model is configurable via env vars without code changes:
    GEN_MODEL_REPO  (default: Qwen/Qwen2.5-1.5B-Instruct)

If the model can't be loaded (no weights cached, no network, OOM) or generation fails
after retries, generation falls back to a deterministic extractive answer built from
the top retrieved chunk instead of crashing the pipeline -- this is the harness's
generation-stage error-recovery path.
"""
from __future__ import annotations

import os
import threading
import time

from harness.schemas import GenerationResult, RetrievalResult

DEFAULT_REPO = os.environ.get("GEN_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct")

_SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant. Answer ONLY using the numbered "
    "context passages given. If the context does not contain the answer, reply exactly: "
    "\"I don't have enough information in the provided context to answer that.\" "
    "Cite the passage numbers you used in square brackets, e.g. [1][2]. Be concise."
)

_load_lock = threading.Lock()


class LocalLLM:
    def __init__(self, repo_id: str = DEFAULT_REPO):
        self.repo_id = repo_id
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None

    def _ensure_loaded(self):
        if self._model is not None or self._load_error is not None:
            return
        with _load_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
            except Exception as e:  # noqa: BLE001 - deliberate broad catch for fallback path
                self._load_error = str(e)

    @staticmethod
    def _build_prompt(query: str, retrieval: RetrievalResult) -> str:
        context_block = "\n".join(
            f"[{i+1}] {rc.chunk.text}" for i, rc in enumerate(retrieval.chunks)
        )
        return (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )

    @staticmethod
    def _extractive_fallback(retrieval: RetrievalResult) -> GenerationResult:
        if not retrieval.chunks:
            return GenerationResult(
                answer="I don't have enough information in the provided context to answer that.",
                citations=[],
                used_fallback=True,
            )
        top = retrieval.chunks[0]
        return GenerationResult(
            answer=f"Based on the most relevant passage: {top.chunk.text}",
            citations=[top.chunk.chunk_id],
            used_fallback=True,
        )

    def generate(self, query: str, retrieval: RetrievalResult, max_tokens: int = 200, max_retries: int = 1) -> GenerationResult:
        t0 = time.perf_counter()
        self._ensure_loaded()

        if self._model is None:
            result = self._extractive_fallback(retrieval)
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        import torch

        user_prompt = self._build_prompt(query, retrieval)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_retries + 1):
            try:
                input_ids = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                )
                with torch.no_grad():
                    output_ids = self._model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0][input_ids.shape[1]:]
                text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                citations = [f"[{i+1}]" for i in range(len(retrieval.chunks)) if f"[{i+1}]" in text]
                return GenerationResult(
                    answer=text,
                    citations=citations or [rc.chunk.chunk_id for rc in retrieval.chunks[:1]],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    retries=attempt,
                )
            except Exception:  # noqa: BLE001
                continue

        result = self._extractive_fallback(retrieval)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.retries = max_retries
        return result
