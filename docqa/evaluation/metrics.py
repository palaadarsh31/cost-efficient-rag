"""Retrieval and answer metrics. Pure functions, unit-tested in tests/test_metrics.py.

Relevance is binary. A question has one or more *facts*; each fact can be found
in one or more chunks. A chunk is relevant to the question if it covers any
fact. Recall@k is measured over facts (did the top-k contain every piece of
evidence the answer needs?), which keeps it honest when overlapping chunks
both contain the same sentence.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], facts: Sequence[set[str]], k: int) -> float:
    if not facts:
        raise ValueError("recall needs at least one fact")
    top = set(ranked[:k])
    return sum(1.0 for fact in facts if fact & top) / len(facts)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for r in ranked[:k] if r in relevant) / k


def context_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Rank-weighted precision (RAGAS 'context precision'): mean of precision@i
    taken at each rank i that holds a relevant chunk. 1.0 means every relevant
    chunk in the top-k sits above every irrelevant one."""
    hits, total = 0, 0.0
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


# ------------------------------------------------------------ answers (SQuAD)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"’", "‘", "“", "”"}


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\[\d+\]", " ", text)  # citation markers are not part of the answer
    text = "".join(ch if ch not in _PUNCT else " " for ch in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str) -> float:
    gold_norm = normalize_answer(gold)
    return float(bool(gold_norm) and gold_norm in normalize_answer(prediction))


# --------------------------------------------------------------- statistics

def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = list(values)
    return float(np.percentile(vals, q)) if vals else None


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 13) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean. With ~25 questions these are wide, on purpose."""
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_resamples, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
