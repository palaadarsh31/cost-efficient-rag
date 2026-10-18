from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# .env is optional; anything already exported in the shell takes precedence.
load_dotenv(override=False)


def _get(name: str, default=None):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    return int(_get(name, default))


def _get_float(name: str, default: float) -> float:
    return float(_get(name, default))


@dataclass
class Settings:
    db_path: Path = Path("data/index/docqa.db")

    # chunking (sizes are in tokens, see chunking.count_tokens)
    chunk_size: int = 350
    chunk_overlap: int = 50

    # embeddings
    embed_backend: str = "fastembed"  # fastembed | openai | hash
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_batch_size: int = 32

    # retrieval
    top_k: int = 5
    retrieval_mode: str = "hybrid"  # hybrid | dense
    min_score: float = 0.5  # cosine similarity floor for the best hit (see the sweep in results/RESULTS.md)

    # generation (any OpenAI-compatible /chat/completions endpoint)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float | None = 0.0  # None: don't send it (some models reject it)
    llm_max_tokens: int = 400
    llm_timeout: float = 120.0

    # judge used by the eval harness; falls back to the generation settings
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_model: str | None = None

    query_log_path: Path | None = Path("logs/queries.jsonl")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.db_path = Path(_get("RAG_DB_PATH", s.db_path))
        s.chunk_size = _get_int("CHUNK_SIZE", s.chunk_size)
        s.chunk_overlap = _get_int("CHUNK_OVERLAP", s.chunk_overlap)

        s.embed_backend = _get("EMBED_BACKEND", s.embed_backend).lower()
        s.embed_model = _get("EMBED_MODEL", s.embed_model)
        # an empty value is meaningful here (no prefix), so read it raw
        if "EMBED_QUERY_PREFIX" in os.environ:
            s.embed_query_prefix = os.environ["EMBED_QUERY_PREFIX"]
        s.embed_base_url = _get("EMBED_BASE_URL", s.embed_base_url)
        s.embed_api_key = _get("EMBED_API_KEY", s.embed_api_key)
        s.embed_batch_size = _get_int("EMBED_BATCH_SIZE", s.embed_batch_size)

        s.top_k = _get_int("TOP_K", s.top_k)
        s.retrieval_mode = _get("RETRIEVAL_MODE", s.retrieval_mode).lower()
        s.min_score = _get_float("MIN_SCORE", s.min_score)

        s.llm_base_url = _get("LLM_BASE_URL", s.llm_base_url)
        s.llm_api_key = _get("LLM_API_KEY", _get("OPENAI_API_KEY"))
        s.llm_model = _get("LLM_MODEL", s.llm_model)
        temperature = _get("LLM_TEMPERATURE", "0")
        s.llm_temperature = None if temperature.lower() in ("none", "off") else float(temperature)
        s.llm_max_tokens = _get_int("LLM_MAX_TOKENS", s.llm_max_tokens)
        s.llm_timeout = _get_float("LLM_TIMEOUT", s.llm_timeout)

        s.judge_base_url = _get("JUDGE_BASE_URL", s.llm_base_url)
        s.judge_api_key = _get("JUDGE_API_KEY", s.llm_api_key)
        s.judge_model = _get("JUDGE_MODEL", s.llm_model)

        log_path = _get("QUERY_LOG_PATH", str(s.query_log_path))
        s.query_log_path = None if log_path.lower() in ("none", "off", "-") else Path(log_path)
        s.log_level = _get("LOG_LEVEL", s.log_level).upper()
        s.validate()
        return s

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.embed_backend not in ("fastembed", "openai", "hash"):
            raise ValueError(f"unknown EMBED_BACKEND {self.embed_backend!r}")
        if self.retrieval_mode not in ("dense", "hybrid"):
            raise ValueError(f"unknown RETRIEVAL_MODE {self.retrieval_mode!r}")
        if self.top_k < 1:
            raise ValueError("TOP_K must be >= 1")

    def redacted(self) -> dict:
        """Settings as a dict that is safe to print or log."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if "api_key" in f.name:
                value = "***" if value else None
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out

    def copy(self, **changes) -> "Settings":
        data = asdict(self)
        data.update(changes)
        return Settings(**data)
