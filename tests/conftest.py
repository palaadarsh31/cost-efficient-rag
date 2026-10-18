from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from docqa.config import Settings
from docqa.embeddings import HashEmbedder
from docqa.llm import Completion

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


class FakeLLM:
    """Records calls and replies with a canned answer."""

    model = "fake-llm"

    def __init__(self, reply: str = "Config lives in environment variables [1]."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages, **_):
        self.calls.append(messages)
        return Completion(text=self.reply, model=self.model, prompt_tokens=120,
                          completion_tokens=12, latency_ms=5.0)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(db_path=tmp_path / "index.db", embed_backend="hash", chunk_size=120,
                    chunk_overlap=20, min_score=0.05, query_log_path=tmp_path / "queries.jsonl")


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture
def small_corpus(tmp_path) -> Path:
    """A few real documents copied into a scratch dir so tests can modify them."""
    root = tmp_path / "corpus"
    for rel in ("twelve-factor/config.md", "twelve-factor/logs.md", "code-review/reviewer-speed.html",
                "prometheus/practices-pushing.md"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CORPUS / rel, target)
    return root
