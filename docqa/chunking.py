"""Structure-aware chunking with exact character offsets.

Blocks from the loader are broken into units (whole blocks if they fit,
otherwise sentences, otherwise fixed token windows) and greedily packed into
chunks of at most `chunk_size` tokens. Overlap is carried over in whole units,
so a chunk never starts in the middle of a sentence unless the sentence itself
was too long to fit.

"Tokens" here are regex word/punctuation units. For English prose the BGE
WordPiece tokenizer produces roughly 1.1-1.3x as many, so the 350 default stays
under the model's 512-token window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


@dataclass
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int
    end: int
    n_tokens: int
    section: str | None
    page_start: int | None
    page_end: int | None


@dataclass
class _Unit:
    start: int
    end: int
    tokens: int
    section: str | None
    page: int | None
    heading: bool = False


def _split_long(text: str, offset: int, limit: int, overlap: int) -> list[tuple[int, int]]:
    """Split one oversized block into (start, end) spans of at most `limit` tokens."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _SENTENCE_END.finditer(text):
        # The separator can begin with closing quotes/brackets. They belong to
        # the preceding sentence; only separator whitespace may be skipped.
        spans.append((pos, match.start() + len(match.group().rstrip())))
        pos = match.end()
    spans.append((pos, len(text)))

    out: list[tuple[int, int]] = []
    for s, e in spans:
        piece = text[s:e]
        if not piece.strip():
            continue
        tokens = list(TOKEN_RE.finditer(piece))
        if len(tokens) <= limit:
            out.append((offset + s, offset + e))
            continue
        # a single "sentence" longer than the chunk (tables, code, run-on lists)
        step = max(1, limit - overlap)
        for i in range(0, len(tokens), step):
            window = tokens[i:i + limit]
            out.append((offset + s + window[0].start(), offset + s + window[-1].end()))
            if i + limit >= len(tokens):
                break
    return out


def _units(doc: Document, chunk_size: int, overlap: int) -> list[_Unit]:
    units: list[_Unit] = []
    for block in doc.blocks:
        n = count_tokens(block.text)
        if n <= chunk_size:
            units.append(_Unit(block.start, block.end, n, block.section, block.page, block.kind == "heading"))
            continue
        for s, e in _split_long(block.text, block.start, chunk_size, overlap):
            units.append(_Unit(s, e, count_tokens(doc.text[s:e]), block.section, block.page))
    return units


def chunk_document(doc: Document, chunk_size: int = 350, overlap: int = 50) -> list[Chunk]:
    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be nonnegative and smaller than chunk_size")

    units = _units(doc, chunk_size, overlap)
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0

    def span_tokens(group: list[_Unit]) -> int:
        # Fixed token windows can overlap. Their sum is not the token cost of
        # the exact source span that will actually be stored and embedded.
        return count_tokens(doc.text[group[0].start:group[-1].end]) if group else 0

    def close(carry_overlap: bool) -> None:
        nonlocal current, current_tokens
        # don't let a chunk end on a dangling heading; push it to the next chunk
        trailing: list[_Unit] = []
        while current and current[-1].heading:
            trailing.insert(0, current.pop())
        if current:
            groups.append(current)
        carried: list[_Unit] = []
        if carry_overlap and current and overlap > 0:
            budget = overlap
            for unit in reversed(current):
                if unit.tokens > budget or unit.heading:
                    break
                carried.insert(0, unit)
                budget -= unit.tokens
            # carrying the whole previous chunk would make no progress
            if len(carried) == len(current):
                carried = []
        current = carried + trailing
        current_tokens = span_tokens(current)

    for unit in units:
        if unit.heading and current_tokens >= chunk_size // 2:
            close(carry_overlap=False)  # new section: start fresh, no overlap
        elif current and span_tokens(current + [unit]) > chunk_size:
            close(carry_overlap=True)
            # overlap + this unit may still not fit; drop the overlap then
            if span_tokens(current + [unit]) > chunk_size:
                current = [u for u in current if u.heading]
                current_tokens = span_tokens(current)
        current.append(unit)
        current_tokens = span_tokens(current)
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for group in groups:
        start, end = group[0].start, group[-1].end
        spans = [(start, end)]
        if count_tokens(doc.text[start:end]) > chunk_size:
            # A carried heading plus a full-size body unit can exceed the
            # budget even after overlap is dropped. Split only those spans,
            # retaining every token and exact source offsets.
            tokens = list(TOKEN_RE.finditer(doc.text, start, end))
            spans = []
            for i in range(0, len(tokens), chunk_size - overlap):
                window = tokens[i:i + chunk_size]
                spans.append((start if i == 0 else window[0].start(),
                              end if i + chunk_size >= len(tokens) else window[-1].end()))
                if i + chunk_size >= len(tokens):
                    break
        for start, end in spans:
            text = doc.text[start:end]
            covered = [u for u in group if u.start < end and u.end > start]
            pages = [u.page for u in covered if u.page is not None]
            body = next((u for u in covered if not u.heading), covered[0])
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    index=len(chunks),
                    text=text,
                    start=start,
                    end=end,
                    n_tokens=count_tokens(text),
                    section=body.section,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                )
            )
    return chunks
