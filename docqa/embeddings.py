from __future__ import annotations

import hashlib
import re
import time
from typing import Protocol, Sequence

import numpy as np
import requests

from .config import Settings


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (CPU, no torch)."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32):
        from fastembed import TextEmbedding  # imported lazily, it's a heavy import

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)
        self.dim = int(len(next(iter(self._model.embed(["dimension probe"])))))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._model.embed(list(texts), batch_size=self.batch_size))
        return _normalize(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._model.embed([self.query_prefix + text])))
        return _normalize(vector)

    def audit_documents(self, texts: Sequence[str]) -> dict:
        """Measure tokenizer truncation on the exact strings sent for embedding.

        The tokenizer is copied before disabling truncation/padding, so this
        diagnostic never changes the embedding model or its token settings.
        Callers must include the same title/section headers used at ingestion.
        """
        inner = getattr(self._model, "model", self._model)
        tokenizer = getattr(inner, "tokenizer", None)
        return audit_tokenizer_inputs(tokenizer, texts, self.model_name)


def audit_tokenizer_inputs(tokenizer, texts: Sequence[str], model_name: str) -> dict:
    """Inspect a loaded Hugging Face tokenizer without loading/downloading a model."""
    if tokenizer is None or not callable(getattr(tokenizer, "to_str", None)):
        return {"available": False, "model": model_name,
                "reason": "loaded embedding backend does not expose a serializable tokenizer"}
    from tokenizers import Tokenizer

    configuration = tokenizer.truncation
    max_length = configuration.get("max_length") if configuration else None
    unlimited = Tokenizer.from_str(tokenizer.to_str())
    unlimited.no_truncation()
    unlimited.no_padding()
    counts = [len(encoding.ids) for encoding in unlimited.encode_batch(list(texts))]
    per_input = [{"index": index, "tokens_including_special": count,
                  "tokens_discarded": max(0, count - max_length) if max_length else 0,
                  "truncated": bool(max_length and count > max_length)}
                 for index, count in enumerate(counts)]
    return {"available": True, "model": model_name, "scope": "tokenizer truncation",
            "n_inputs": len(counts), "max_length": max_length,
            "truncation_configuration": configuration,
            "max_input_tokens": max(counts) if counts else None,
            "n_truncated": sum(row["truncated"] for row in per_input),
            "tokens_discarded": sum(row["tokens_discarded"] for row in per_input),
            "per_input": per_input}


class OpenAIEmbedder:
    """Any server that speaks the OpenAI /embeddings API (OpenAI, Ollama, vLLM, ...)."""

    def __init__(self, model_name: str, base_url: str, api_key: str | None = None,
                 query_prefix: str = "", batch_size: int = 64, timeout: float = 60.0):
        self.model_name = model_name
        self.url = base_url.rstrip("/") + "/embeddings"
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.dim = int(self._call(["dimension probe"]).shape[1])

    def _call(self, texts: list[str]) -> np.ndarray:
        for attempt in range(5):
            resp = self.session.post(self.url, json={"model": self.model_name, "input": texts},
                                     timeout=self.timeout)
            if resp.status_code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            return _normalize(np.array([d["embedding"] for d in data]))
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [self._call(list(texts[i:i + self.batch_size]))
                 for i in range(0, len(texts), self.batch_size)]
        return np.vstack(parts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([self.query_prefix + text])[0]


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how i in is it of on or that the this to "
    "was what when where which who why will with you your do does can".split()
)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Not a semantic model: it exists so tests and offline smoke runs work without
    downloading anything. Never use it for the reported numbers.
    """

    def __init__(self, dim: int = 384):
        self.model_name = f"hash-bow-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feat in features:
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text))


def build_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "fastembed":
        return FastEmbedEmbedder(settings.embed_model, settings.embed_query_prefix,
                                 settings.embed_batch_size)
    if settings.embed_backend == "openai":
        if not settings.embed_base_url:
            raise ValueError("EMBED_BASE_URL is required for EMBED_BACKEND=openai")
        return OpenAIEmbedder(settings.embed_model, settings.embed_base_url,
                              settings.embed_api_key, settings.embed_query_prefix)
    return HashEmbedder()
