"""Idempotent ingestion.

A document is identified by its path relative to the corpus root. On every run
we hash the file bytes; if both the hash and the chunking/embedding config
fingerprint match what is stored, the file is skipped without being parsed or
embedded. Otherwise its old chunks are replaced in a single transaction. Chunk
ids are unique in the schema, so even a buggy caller cannot insert duplicates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_document
from .config import Settings
from .embeddings import Embedder
from .loaders import SUPPORTED_SUFFIXES, Document, file_sha256, load_document
from .store import VectorStore

log = logging.getLogger(__name__)

CHUNKER_VERSION = 2


@dataclass
class IngestReport:
    root: str
    files_seen: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks_written: int = 0
    vectors_before: int = 0
    vectors_after: int = 0
    seconds: float = 0.0
    failed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def config_fingerprint(settings: Settings, embedder: Embedder) -> str:
    payload = {
        "chunker": CHUNKER_VERSION,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embed_model": embedder.model_name,
        "dim": embedder.dim,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def embedding_text(doc: Document, chunk: Chunk) -> str:
    """What actually gets embedded: a short title/section header plus the chunk.

    The stored chunk text stays a verbatim span of the document; the header only
    helps short chunks that don't repeat what they're about.
    """
    header = [doc.title]
    if chunk.section and chunk.section not in doc.title:
        header.append(chunk.section)
    return " | ".join(header) + "\n\n" + chunk.text


def iter_corpus_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for p in path.rglob("*"):
        rel_parts = p.relative_to(path).parts
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not any(
            part.startswith(".") for part in rel_parts
        ):
            files.append(p)
    return sorted(files)


def _resolve_root(store: VectorStore, path: Path) -> Path:
    """Keep doc ids stable: reuse the root recorded on the first ingest when possible."""
    path = path.resolve()
    recorded = store.meta().get("corpus_root")
    if recorded:
        root = Path(recorded)
        if path == root or root in path.parents:
            return root
        log.warning("ingesting %s outside the recorded corpus root %s; ids are relative to %s",
                    path, root, path)
    root = path if path.is_dir() else path.parent
    if not recorded:
        with store.lock, store.db:
            store.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('corpus_root', ?)",
                             (str(root),))
    return root


def ingest_path(path: str | Path, store: VectorStore, embedder: Embedder, settings: Settings,
                prune: bool = False) -> IngestReport:
    started = time.perf_counter()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    store.open_index(embedder.dim, embedder.model_name)
    root = _resolve_root(store, path)
    fingerprint = config_fingerprint(settings, embedder)
    report = IngestReport(root=str(root), vectors_before=store.counts()["vectors"])

    seen: set[str] = set()
    for file in iter_corpus_files(path.resolve()):
        report.files_seen += 1
        doc_id = file.relative_to(root).as_posix()
        seen.add(doc_id)
        try:
            existing = store.get_document(doc_id)
            if (existing is not None and existing["content_hash"] == file_sha256(file)
                    and existing["config_fingerprint"] == fingerprint):
                report.unchanged += 1
                continue

            doc = load_document(file, root)
            chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
            vectors = embedder.embed_documents([embedding_text(doc, c) for c in chunks])
            store.replace_document(doc, chunks, vectors, fingerprint)
            report.chunks_written += len(chunks)
            if existing is None:
                report.added += 1
                log.info("added %s (%d chunks)", doc_id, len(chunks))
            else:
                report.updated += 1
                log.info("updated %s (%d chunks)", doc_id, len(chunks))
        except Exception as exc:  # one bad file shouldn't sink the whole run
            log.exception("failed to ingest %s", doc_id)
            report.failed.append(f"{doc_id}: {exc}")

    if prune:
        prefix = path.resolve().relative_to(root).as_posix() if path.resolve() != root else ""
        for doc_id in store.doc_ids():
            in_scope = not prefix or doc_id == prefix or doc_id.startswith(prefix.rstrip("/") + "/")
            if in_scope and doc_id not in seen:
                store.delete_document(doc_id)
                report.removed += 1
                log.info("removed %s (file no longer present)", doc_id)

    report.vectors_after = store.counts()["vectors"]
    report.seconds = round(time.perf_counter() - started, 3)
    return report
