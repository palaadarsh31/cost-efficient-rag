"""HTTP API.

Run with:  uvicorn docqa.api:create_app --factory --port 8000
       or: python -m docqa serve
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .ingest import ingest_path
from .llm import LLMError
from .qa import build_pipeline

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["dense", "hybrid"] | None = None
    filters: dict[str, str] | None = Field(
        default=None,
        description="Equality filters on doc_type, collection or doc_id",
        examples=[{"collection": "prometheus"}],
    )


class IngestRequest(BaseModel):
    path: str = "data/corpus"
    prune: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = build_pipeline(settings)
    store = pipeline.retriever.store

    app = FastAPI(title="docqa", version=__version__,
                  description="Grounded question answering over a document corpus (SQLite + sqlite-vec).")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "index": store.counts(),
            "embed_model": pipeline.retriever.embedder.model_name,
            "embed_dim": pipeline.retriever.embedder.dim,
            "llm_model": settings.llm_model,
            "retrieval_mode": settings.retrieval_mode,
            "top_k": settings.top_k,
        }

    @app.post("/query")
    def query(req: QueryRequest):
        try:
            result = pipeline.answer(req.question, k=req.k, filters=req.filters, mode=req.mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except LLMError as exc:
            log.error("LLM call failed: %s", exc)
            raise HTTPException(status_code=502, detail="the language model call failed; see server log")
        return result.as_dict()

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        try:
            report = ingest_path(req.path, store, pipeline.retriever.embedder, settings, prune=req.prune)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"path not found: {req.path}")
        return report.as_dict()

    return app
