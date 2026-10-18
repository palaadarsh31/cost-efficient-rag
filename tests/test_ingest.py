import pytest

from docqa.embeddings import HashEmbedder
from docqa.ingest import ingest_path
from docqa.store import IndexMismatchError, VectorStore


def _consistent(store):
    c = store.counts()
    assert c["chunks"] == c["vectors"] == c["fts_rows"]
    return c


def test_reingest_is_a_no_op(small_corpus, settings, embedder):
    store = VectorStore(settings.db_path)
    first = ingest_path(small_corpus, store, embedder, settings)
    before = _consistent(store)
    assert first.added == 4 and before["vectors"] > 0

    second = ingest_path(small_corpus, store, embedder, settings)
    after = _consistent(store)
    assert second.unchanged == 4
    assert second.added == second.updated == second.chunks_written == 0
    assert after == before
    assert second.vectors_before == second.vectors_after == before["vectors"]


def test_changed_file_replaces_only_its_chunks(small_corpus, settings, embedder):
    store = VectorStore(settings.db_path)
    ingest_path(small_corpus, store, embedder, settings)
    logs_before = [c.chunk_id for c in store.chunks_for("twelve-factor/logs.md")]
    config_before = [c.chunk_id for c in store.chunks_for("twelve-factor/config.md")]

    path = small_corpus / "twelve-factor/config.md"
    path.write_text(path.read_text() + "\n\nOne more paragraph about config that was added later.\n")
    report = ingest_path(small_corpus, store, embedder, settings)

    assert (report.updated, report.unchanged, report.added) == (1, 3, 0)
    assert [c.chunk_id for c in store.chunks_for("twelve-factor/logs.md")] == logs_before
    assert [c.chunk_id for c in store.chunks_for("twelve-factor/config.md")] != config_before
    _consistent(store)


def test_prune_removes_deleted_files(small_corpus, settings, embedder):
    store = VectorStore(settings.db_path)
    ingest_path(small_corpus, store, embedder, settings)
    (small_corpus / "prometheus/practices-pushing.md").unlink()

    kept = ingest_path(small_corpus, store, embedder, settings)  # without --prune nothing is dropped
    assert kept.removed == 0 and "prometheus/practices-pushing.md" in store.doc_ids()

    pruned = ingest_path(small_corpus, store, embedder, settings, prune=True)
    assert pruned.removed == 1
    assert "prometheus/practices-pushing.md" not in store.doc_ids()
    _consistent(store)


def test_ingesting_a_subfolder_keeps_doc_ids_stable(small_corpus, settings, embedder):
    store = VectorStore(settings.db_path)
    ingest_path(small_corpus, store, embedder, settings)
    n = store.counts()["vectors"]
    report = ingest_path(small_corpus / "twelve-factor", store, embedder, settings)
    assert report.unchanged == 2 and report.added == 0
    assert store.counts()["vectors"] == n


def test_config_change_triggers_rechunk(small_corpus, settings, embedder):
    store = VectorStore(settings.db_path)
    ingest_path(small_corpus, store, embedder, settings)
    report = ingest_path(small_corpus, store, embedder, settings.copy(chunk_size=80, chunk_overlap=10))
    assert report.updated == 4
    _consistent(store)


def test_embedder_mismatch_is_refused(small_corpus, settings, embedder):
    store = VectorStore(settings.db_path)
    ingest_path(small_corpus, store, embedder, settings)
    with pytest.raises(IndexMismatchError):
        ingest_path(small_corpus, store, HashEmbedder(dim=128), settings)
