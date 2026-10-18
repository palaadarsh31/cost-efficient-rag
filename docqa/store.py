"""SQLite storage: chunk text + metadata, sqlite-vec vectors, FTS5 keyword index.

Everything lives in one file. `chunks.id` is the rowid shared by the vec0 and
FTS5 virtual tables, so a chunk and its vector/keyword entries are always
written and deleted together inside one transaction.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sqlite_vec

from .chunking import Chunk
from .loaders import Document

SCHEMA_VERSION = "1"
FILTER_COLUMNS = ("doc_type", "collection", "doc_id")

_FTS_TOKEN = re.compile(r"\w+")
_FTS_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i in is it of on or "
    "should that the this to was what when where which who why will with".split()
)


class IndexMismatchError(RuntimeError):
    pass


@dataclass
class StoredChunk:
    rowid: int
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    start_char: int
    end_char: int
    n_tokens: int
    title: str
    doc_type: str
    collection: str


def make_chunk_id(doc_id: str, chunk: Chunk) -> str:
    raw = f"{doc_id}\x1f{chunk.index}\x1f{chunk.text}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _check_filters(filters: dict | None) -> dict:
    filters = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
    unknown = set(filters) - set(FILTER_COLUMNS)
    if unknown:
        raise ValueError(f"unsupported filter(s): {sorted(unknown)}; allowed: {FILTER_COLUMNS}")
    return filters


class VectorStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")
            self.db.execute("PRAGMA synchronous = NORMAL")
        self.lock = threading.RLock()
        self.dim: int | None = None
        self.model_name: str | None = None

    # ------------------------------------------------------------- schema
    def open_index(self, dim: int, model_name: str) -> None:
        """Create the schema, or check that an existing index matches the embedder."""
        with self.lock, self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            meta = dict(self.db.execute("SELECT key, value FROM meta").fetchall())
            if meta:
                if meta.get("embed_model") != model_name or int(meta.get("embed_dim", -1)) != dim:
                    raise IndexMismatchError(
                        f"index at {self.path} was built with {meta.get('embed_model')} "
                        f"({meta.get('embed_dim')}d), current embedder is {model_name} ({dim}d). "
                        "Re-ingest with --rebuild or point RAG_DB_PATH somewhere else."
                    )
            else:
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.db.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [("schema_version", SCHEMA_VERSION), ("embed_model", model_name),
                     ("embed_dim", str(dim)), ("created_at", now)],
                )
            self.db.executescript(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    doc_type TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    title TEXT,
                    content_hash TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    n_chunks INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    n_tokens INTEGER NOT NULL,
                    UNIQUE (doc_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    embedding float[{dim}] distance_metric=cosine,
                    doc_type text,
                    collection text,
                    doc_id text
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    text, title, section, tokenize = 'porter unicode61'
                );
            """)
        self.dim, self.model_name = dim, model_name

    def meta(self) -> dict:
        with self.lock:
            try:
                return dict(self.db.execute("SELECT key, value FROM meta").fetchall())
            except sqlite3.OperationalError:
                return {}

    # ------------------------------------------------------------- writes
    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        with self.lock:
            return self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()

    def doc_ids(self) -> list[str]:
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT doc_id FROM documents ORDER BY doc_id")]

    def _delete_chunks(self, doc_id: str) -> int:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
        for rowid in ids:
            self.db.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
            self.db.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rowid,))
        self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return len(ids)

    def delete_document(self, doc_id: str) -> int:
        with self.lock, self.db:
            removed = self._delete_chunks(doc_id)
            self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return removed

    def replace_document(self, doc: Document, chunks: list[Chunk], vectors: np.ndarray,
                         fingerprint: str) -> None:
        """Atomically swap a document's chunks for a new set (insert if new)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if len(vectors) and vectors.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-d vectors, got {vectors.shape[1]}")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock, self.db:
            self._delete_chunks(doc.doc_id)
            self.db.execute(
                """INSERT INTO documents (doc_id, doc_type, collection, title, content_hash,
                                          config_fingerprint, n_chunks, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                       doc_type = excluded.doc_type, collection = excluded.collection,
                       title = excluded.title, content_hash = excluded.content_hash,
                       config_fingerprint = excluded.config_fingerprint,
                       n_chunks = excluded.n_chunks, ingested_at = excluded.ingested_at""",
                (doc.doc_id, doc.doc_type, doc.collection, doc.title, doc.content_hash,
                 fingerprint, len(chunks), now),
            )
            for chunk, vector in zip(chunks, vectors):
                cur = self.db.execute(
                    """INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, section, page_start,
                                           page_end, start_char, end_char, n_tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (make_chunk_id(doc.doc_id, chunk), doc.doc_id, chunk.index, chunk.text,
                     chunk.section, chunk.page_start, chunk.page_end, chunk.start, chunk.end,
                     chunk.n_tokens),
                )
                rowid = cur.lastrowid
                self.db.execute(
                    "INSERT INTO vec_chunks (rowid, embedding, doc_type, collection, doc_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rowid, np.asarray(vector, dtype=np.float32).tobytes(), doc.doc_type,
                     doc.collection, doc.doc_id),
                )
                self.db.execute(
                    "INSERT INTO fts_chunks (rowid, text, title, section) VALUES (?, ?, ?, ?)",
                    (rowid, chunk.text, doc.title, chunk.section or ""),
                )

    # -------------------------------------------------------------- reads
    def search_dense(self, query_vector: np.ndarray, k: int,
                     filters: dict | None = None) -> list[tuple[int, float]]:
        """Top-k by cosine distance. Returns (rowid, similarity) best first."""
        filters = _check_filters(filters)
        sql = "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?"
        params: list = [np.asarray(query_vector, dtype=np.float32).tobytes(), k]
        for column, value in filters.items():
            sql += f" AND {column} = ?"
            params.append(value)
        sql += " ORDER BY distance"
        with self.lock:
            rows = self.db.execute(sql, params).fetchall()
        return [(r["rowid"], 1.0 - float(r["distance"])) for r in rows]

    def search_keyword(self, query: str, k: int,
                       filters: dict | None = None) -> list[tuple[int, float]]:
        """Top-k BM25 matches from FTS5. Returns (rowid, bm25) best first (lower bm25 is better)."""
        filters = _check_filters(filters)
        terms = [t for t in _FTS_TOKEN.findall(query.lower()) if t not in _FTS_STOP]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        sql = ("SELECT f.rowid AS rowid, bm25(fts_chunks) AS score FROM fts_chunks f "
               "JOIN chunks c ON c.id = f.rowid JOIN documents d ON d.doc_id = c.doc_id "
               "WHERE fts_chunks MATCH ?")
        params: list = [match]
        for column, value in filters.items():
            sql += f" AND {'c' if column == 'doc_id' else 'd'}.{column} = ?"
            params.append(value)
        sql += " ORDER BY score LIMIT ?"
        params.append(k)
        with self.lock:
            rows = self.db.execute(sql, params).fetchall()
        return [(r["rowid"], float(r["score"])) for r in rows]

    def fetch(self, rowids: list[int]) -> dict[int, StoredChunk]:
        if not rowids:
            return {}
        marks = ",".join("?" * len(rowids))
        with self.lock:
            rows = self.db.execute(
                f"""SELECT c.id AS rowid, c.chunk_id, c.doc_id, c.chunk_index, c.text, c.section,
                           c.page_start, c.page_end, c.start_char, c.end_char, c.n_tokens,
                           d.title, d.doc_type, d.collection
                    FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                    WHERE c.id IN ({marks})""",
                list(rowids),
            ).fetchall()
        return {r["rowid"]: StoredChunk(**dict(r)) for r in rows}

    def chunks_for(self, doc_id: str) -> list[StoredChunk]:
        with self.lock:
            ids = [r[0] for r in self.db.execute(
                "SELECT id FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))]
        found = self.fetch(ids)
        return [found[i] for i in ids]

    def counts(self) -> dict:
        with self.lock:
            try:
                one = lambda sql: self.db.execute(sql).fetchone()[0]
                return {
                    "documents": one("SELECT count(*) FROM documents"),
                    "chunks": one("SELECT count(*) FROM chunks"),
                    "vectors": one("SELECT count(*) FROM vec_chunks"),
                    "fts_rows": one("SELECT count(*) FROM fts_chunks"),
                }
            except sqlite3.OperationalError:
                return {"documents": 0, "chunks": 0, "vectors": 0, "fts_rows": 0}

    def size_bytes(self) -> int:
        if str(self.path) == ":memory:":
            return 0
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*"))

    def close(self) -> None:
        with self.lock:
            self.db.close()
