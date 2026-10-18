"""Gold questions and chunk-level relevance labels.

Questions carry verbatim evidence quotes instead of chunk ids, so the same
question set works for any chunking config. A chunk counts as covering a quote
when it contains at least `min_overlap` of the quote's characters.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..store import VectorStore


@dataclass
class Question:
    id: str
    question: str
    answer: str | None
    answerable: bool
    evidence: list[dict] = field(default_factory=list)
    note: str | None = None


@dataclass
class Labels:
    facts: list[set[str]]  # per fact: chunk_ids that cover it
    relevant: set[str]  # union of the above


def load_questions(path: str | Path) -> list[Question]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                rows.append(Question(**{k: data.get(k) for k in Question.__dataclass_fields__}
                                     | {"evidence": data.get("evidence") or []}))
    ids = [q.id for q in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return rows


def _find_span(text: str, quote: str) -> tuple[int, int] | None:
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    # tolerate whitespace differences (e.g. a quote that crosses a line break)
    pattern = r"\s+".join(re.escape(tok) for tok in quote.split())
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def label_question(question: Question, store: VectorStore, doc_texts: dict[str, str],
                   min_overlap: float = 0.5) -> Labels:
    facts: list[set[str]] = []
    for item in question.evidence:
        locations = [item] + list(item.get("alt", []))
        covering: set[str] = set()
        for loc in locations:
            text = doc_texts.get(loc["doc_id"])
            if text is None:
                raise KeyError(f"{question.id}: {loc['doc_id']} is not in the index")
            span = _find_span(text, loc["quote"])
            if span is None:
                raise ValueError(f"{question.id}: evidence quote not found in {loc['doc_id']}: "
                                 f"{loc['quote'][:60]!r}")
            q_start, q_end = span
            for chunk in store.chunks_for(loc["doc_id"]):
                overlap = min(q_end, chunk.end_char) - max(q_start, chunk.start_char)
                if overlap > 0 and overlap >= min_overlap * (q_end - q_start):
                    covering.add(chunk.chunk_id)
        if not covering:
            raise ValueError(f"{question.id}: no chunk covers the evidence; check the chunker")
        facts.append(covering)
    relevant = set().union(*facts) if facts else set()
    return Labels(facts=facts, relevant=relevant)
