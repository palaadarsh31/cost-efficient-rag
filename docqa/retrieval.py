from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .embeddings import Embedder
from .store import StoredChunk, VectorStore

RRF_K = 60  # standard constant from the reciprocal rank fusion paper


@dataclass
class Hit:
    rank: int
    chunk: StoredChunk
    score: float  # cosine similarity (dense) or fused RRF score (hybrid)
    similarity: float | None  # dense cosine similarity when known


@dataclass
class Retrieval:
    query: str
    mode: str
    k: int
    hits: list[Hit]
    top_similarity: float | None
    timings_ms: dict = field(default_factory=dict)


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, rowid in enumerate(ranking, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def retrieve(self, query: str, k: int | None = None, filters: dict | None = None,
                 mode: str | None = None) -> Retrieval:
        k = k or self.settings.top_k
        mode = (mode or self.settings.retrieval_mode).lower()
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        t1 = time.perf_counter()

        if mode == "dense":
            dense = self.store.search_dense(query_vector, k, filters)
            ranked = dense
        elif mode == "hybrid":
            pool = max(4 * k, 20)
            dense = self.store.search_dense(query_vector, pool, filters)
            keyword = self.store.search_keyword(query, pool, filters)
            ranked = rrf_fuse([[r for r, _ in dense], [r for r, _ in keyword]])[:k]
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        t2 = time.perf_counter()

        similarity = dict(dense)
        chunks = self.store.fetch([rowid for rowid, _ in ranked])
        hits = [
            Hit(rank=i, chunk=chunks[rowid], score=round(score, 6),
                similarity=round(similarity[rowid], 6) if rowid in similarity else None)
            for i, (rowid, score) in enumerate(ranked, start=1)
            if rowid in chunks
        ]
        t3 = time.perf_counter()

        return Retrieval(
            query=query,
            mode=mode,
            k=k,
            hits=hits,
            top_similarity=round(dense[0][1], 6) if dense else None,
            timings_ms={
                "embed": round((t1 - t0) * 1000, 3),
                "search": round((t2 - t1) * 1000, 3),
                "fetch": round((t3 - t2) * 1000, 3),
                "total": round((t3 - t0) * 1000, 3),
            },
        )
