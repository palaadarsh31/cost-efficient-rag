"""Turn PDF / HTML / Markdown files into a flat list of text blocks.

Every loader produces the same shape: an ordered list of blocks (headings and
paragraph-ish units) that are joined with blank lines into the document text.
Block offsets into that text are kept so chunks can be traced back to the
exact span they came from (the eval harness relies on this).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".markdown": "md",
}

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")


def clean_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _inline_text(node) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", clean_ws(node.get_text()))


@dataclass
class Block:
    text: str
    kind: str = "text"  # "text" | "heading"
    section: str | None = None
    page: int | None = None
    start: int = 0
    end: int = 0


@dataclass
class Document:
    doc_id: str  # path relative to the corpus root, posix style
    path: Path
    doc_type: str
    collection: str
    title: str
    content_hash: str
    blocks: list[Block] = field(default_factory=list)
    text: str = ""

    def finalize(self) -> "Document":
        """Join blocks into the document text and record each block's offsets."""
        parts, pos = [], 0
        for block in self.blocks:
            if parts:
                pos += 2
            block.start = pos
            block.end = pos + len(block.text)
            parts.append(block.text)
            pos = block.end
        self.text = "\n\n".join(parts)
        return self


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for piece in iter(lambda: fh.read(1 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def load_document(path: Path, root: Path) -> Document:
    path = Path(path)
    doc_type = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if doc_type is None:
        raise ValueError(f"unsupported file type: {path}")

    rel = path.resolve().relative_to(Path(root).resolve())
    collection = rel.parts[0] if len(rel.parts) > 1 else "default"

    if doc_type == "pdf":
        title, blocks = _parse_pdf(path)
    elif doc_type == "html":
        title, blocks = _parse_html(path.read_text(encoding="utf-8", errors="replace"))
    else:
        title, blocks = _parse_markdown(path.read_text(encoding="utf-8", errors="replace"))

    doc = Document(
        doc_id=rel.as_posix(),
        path=path,
        doc_type=doc_type,
        collection=collection,
        title=title or path.stem.replace("-", " ").replace("_", " "),
        content_hash=file_sha256(path),
        blocks=[b for b in blocks if b.text],
    )
    return doc.finalize()


# --------------------------------------------------------------------- HTML

_DROP_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "svg", "button", "iframe", "template", "select",
}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_LEAF_BLOCKS = {"p", "pre", "dt", "dd", "figcaption", "caption", "summary"}
_INLINE = {
    "a", "abbr", "b", "br", "cite", "code", "em", "i", "img", "kbd", "mark",
    "q", "s", "samp", "small", "span", "strong", "sub", "sup", "time", "u", "var",
}


def _parse_html(html: str) -> tuple[str, list[Block]]:
    from bs4 import BeautifulSoup, Comment, NavigableString

    soup = BeautifulSoup(html, "lxml")
    title = clean_ws(soup.title.get_text()) if soup.title else ""

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(attrs={"role": ["navigation", "banner", "contentinfo"]}):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[Block] = []
    headings: dict[int, str] = {}
    pending: list[str] = []

    def section() -> str | None:
        return " > ".join(headings[k] for k in sorted(headings)) or None

    def flush():
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", clean_ws("".join(pending)))
        pending.clear()
        if text:
            blocks.append(Block(text, section=section()))

    def emit(text: str):
        text = clean_ws(text)
        if text:
            blocks.append(Block(text, section=section()))

    def walk(node):
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                if child.strip():
                    pending.append(str(child))
                continue
            name = child.name
            if name in _INLINE:
                pending.append(child.get_text())
            elif name in _HEADINGS:
                flush()
                level = int(name[1])
                text = _inline_text(child)
                if not text:
                    continue
                for deeper in [k for k in headings if k >= level]:
                    del headings[deeper]
                headings[level] = text
                blocks.append(Block(text, kind="heading", section=section()))
            elif name == "table":
                flush()
                for row in child.find_all("tr"):
                    cells = [_inline_text(c) for c in row.find_all(["td", "th"])]
                    emit(" | ".join(c for c in cells if c))
            elif name == "li":
                flush()
                nested = child.find_all(["ul", "ol"], recursive=False)
                for lst in nested:
                    lst.extract()
                emit(_inline_text(child))
                for lst in nested:
                    walk(lst)
            elif name in _LEAF_BLOCKS:
                flush()
                emit(_inline_text(child))
            else:
                # div, section, ul, blockquote, ... : structural, keep descending
                flush()
                walk(child)
        # text left over at the end of a container belongs to that container
        flush()

    walk(root)

    if not title:
        first_heading = next((b.text for b in blocks if b.kind == "heading"), "")
        title = first_heading
    return title, blocks


# ----------------------------------------------------------------- Markdown

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def _parse_markdown(text: str) -> tuple[str, list[Block]]:
    import markdown

    title = ""
    match = _FRONT_MATTER.match(text)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "title":
                title = value.strip().strip("'\"")
        text = text[match.end():]

    html = markdown.markdown(text, extensions=["extra", "sane_lists"])
    html_title, blocks = _parse_html(f"<html><body><main>{html}</main></body></html>")
    return title or html_title, blocks


# ---------------------------------------------------------------------- PDF

_BULLET = re.compile(r"^([•●◦▪‣■□\-–*]|\d{1,2}[.)])\s+")
_TOC_LINE = re.compile(r"\.{5,}\s*\d+\s*$")
_TOC_ENTRY = re.compile(r"^[A-Z(][^.!?]{2,90}\s\d{1,3}$")
_END_PUNCT = (".", "!", "?", ":", ";", "”", '"')


def _parse_pdf(path: Path) -> tuple[str, list[Block]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = clean_ws(str(reader.metadata.title))

    boilerplate = _repeated_edge_lines(pages)
    blocks: list[Block] = []
    current_section: str | None = None

    for page_no, raw in enumerate(pages, start=1):
        # InDesign-style soft hyphens come out as "word -\nrest"
        raw = re.sub(r"(\w) -\n(\w)", r"\1\2", raw)
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [
            ln for ln in lines
            if ln and _norm_edge(ln) not in boilerplate and not _TOC_LINE.search(ln)
        ]
        if sum(1 for ln in lines if _TOC_ENTRY.match(ln)) >= 6:
            lines = [ln for ln in lines if not _TOC_ENTRY.match(ln)]
        for para in _lines_to_paragraphs(lines):
            if _looks_like_heading(para):
                current_section = para
                blocks.append(Block(para, kind="heading", section=current_section, page=page_no))
            else:
                blocks.append(Block(para, section=current_section, page=page_no))

    title = meta_title or _pdf_title_guess(blocks)
    return title, blocks


def _norm_edge(line: str) -> str:
    return re.sub(r"\d+", "#", clean_ws(line).lower())


def _repeated_edge_lines(pages: list[str], edge: int = 2) -> set[str]:
    """Running headers/footers: lines that sit at the top or bottom of many pages."""
    if len(pages) < 4:
        return set()
    counts: Counter[str] = Counter()
    for raw in pages:
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        seen = {_norm_edge(ln) for ln in lines[:edge] + lines[-edge:]}
        counts.update(seen)
    threshold = max(3, int(0.4 * len(pages)))
    return {line for line, n in counts.items() if n >= threshold}


def _lines_to_paragraphs(lines: list[str]) -> list[str]:
    if not lines:
        return []
    typical = sorted(len(ln) for ln in lines)[len(lines) // 2]
    paragraphs: list[str] = []
    current: list[str] = []

    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _BULLET.match(line) and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(line)

        if _looks_like_heading(line) and len(current) == 1:
            paragraphs.append(line)
            current = []
            continue

        ends_sentence = line.endswith(_END_PUNCT)
        short_line = len(line) < 0.75 * typical
        if not nxt or _BULLET.match(nxt) or (ends_sentence and short_line) or _looks_like_heading(nxt):
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))
    return [clean_ws(p) for p in paragraphs if clean_ws(p)]


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    if not 1 <= len(words) <= 8 or len(line) > 70:
        return False
    if line.endswith((".", ",", ";", ":", "?", "!")) or _BULLET.match(line):
        return False
    if not line[0].isupper() or line.replace(" ", "").isdigit():
        return False
    capitalised = sum(1 for w in words if w[0].isupper() or not w[0].isalpha())
    small = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs"}
    return all(w[0].isupper() or w.lower() in small or not w[0].isalpha() for w in words) or (
        len(words) <= 3 and capitalised >= 1
    )


def _pdf_title_guess(blocks: list[Block]) -> str:
    first_page = [b.text for b in blocks if b.page == 1][:4]
    return clean_ws(" ".join(first_page))[:120]
