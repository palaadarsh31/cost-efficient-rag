"""One source representation for generation, judging and controlled replay.

Citation IDs belong to passages, never to their temporary display position.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

FORMAT_VERSION = "source-passages-v1"


@dataclass(frozen=True)
class Passage:
    citation_id: int
    chunk_id: str
    doc_id: str
    title: str
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    def __post_init__(self):
        if type(self.citation_id) is not int or self.citation_id < 1:
            raise ValueError("citation_id must be a positive integer")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("passage text must be nonempty")

    @classmethod
    def from_hit(cls, citation_id: int, hit) -> "Passage":
        c = hit.chunk
        return cls(citation_id, c.chunk_id, c.doc_id, c.title, c.text,
                   c.section, c.page_start, c.page_end)

    @classmethod
    def from_mapping(cls, value: Mapping, fallback_id: int) -> "Passage":
        return cls(citation_id=value.get("citation_id", fallback_id),
                   chunk_id=value.get("chunk_id", ""), doc_id=value.get("doc_id", ""),
                   title=value.get("title", ""), text=value["text"],
                   section=value.get("section"),
                   page_start=value.get("page_start", value.get("page")),
                   page_end=value.get("page_end", value.get("page_start", value.get("page"))))

    def as_dict(self) -> dict:
        return asdict(self)


def coerce_passages(values: Iterable[Passage | Mapping | str]) -> list[Passage]:
    """Legacy text-only callers remain supported, with no invented metadata."""
    out = []
    for n, value in enumerate(values, 1):
        if isinstance(value, Passage):
            passage = value
        elif isinstance(value, str):
            passage = Passage(n, "", "", "", value)
        else:
            passage = Passage.from_mapping(value, n)
        out.append(passage)
    ids = [p.citation_id for p in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate passage citation IDs")
    return out


def format_passage(passage: Passage) -> str:
    where = [passage.doc_id] if passage.doc_id else []
    if passage.page_start is not None:
        end = passage.page_end if passage.page_end is not None else passage.page_start
        where.append(f"p. {passage.page_start}" if passage.page_start == end
                     else f"pp. {passage.page_start}-{end}")
    if passage.section:
        where.append(passage.section)
    heading = passage.title
    if where:
        heading += f" ({'; '.join(where)})"
    return f"[{passage.citation_id}] {heading}\n{passage.text}" if heading else f"[{passage.citation_id}] {passage.text}"


def format_passages(passages: Iterable[Passage | Mapping | str]) -> str:
    return "\n\n".join(format_passage(p) for p in coerce_passages(passages))
