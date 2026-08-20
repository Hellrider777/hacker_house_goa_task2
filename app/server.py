"""FastAPI backend: /query (audio in, harness out) and /query_text (typed fallback),
serving the minimal static frontend from app/frontend/.

The harness is built once at startup (indexes + embedder + LLM loaded and warm) so
per-request latency reflects only the pipeline stages themselves, not cold-start cost.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from harness.build import build_harness
from harness.schemas import PipelineResult

FRONTEND_DIR = Path(__file__).parent / "frontend"
LANGUAGES = os.environ.get("RAG_LANGUAGES", "hi,ta").split(",")

app = FastAPI(title="Voice RAG on MSMARCO-XI")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_harness = None


@app.on_event("startup")
def _load_harness():
    global _harness
    _harness = build_harness(languages=[l.strip() for l in LANGUAGES])


@app.get("/health")
def health():
    return {"status": "ok", "languages": LANGUAGES}


@app.post("/query", response_model=PipelineResult)
async def query_audio(file: UploadFile = File(...), language: str = Form("hi")):
    audio_bytes = await file.read()
    return _harness.run_from_audio(audio_bytes, language=language, filename=file.filename or "audio.wav")


@app.post("/query_text", response_model=PipelineResult)
async def query_text(text: str = Form(""), language: str = Form("hi")):
    return _harness.run_from_text(text, language=language)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
