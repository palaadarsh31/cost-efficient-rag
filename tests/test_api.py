import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # needed by TestClient
from fastapi.testclient import TestClient

from docqa import api
from docqa.embeddings import HashEmbedder
from docqa.qa import QAPipeline
from docqa.retrieval import Retriever
from docqa.store import VectorStore

from conftest import FakeLLM


@pytest.fixture
def client(small_corpus, settings, monkeypatch):
    def fake_build(s, with_llm=True):
        embedder = HashEmbedder(dim=64)
        store = VectorStore(s.db_path)
        store.open_index(embedder.dim, embedder.model_name)
        return QAPipeline(Retriever(store, embedder, s), FakeLLM(), s)

    monkeypatch.setattr(api, "build_pipeline", fake_build)
    app = api.create_app(settings)
    with TestClient(app) as c:
        resp = c.post("/ingest", json={"path": str(small_corpus)})
        assert resp.status_code == 200 and resp.json()["added"] == 4
        yield c


def test_health_reports_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["index"]["vectors"] > 0


def test_query_returns_cited_answer(client):
    resp = client.post("/query", json={"question": "Where is config stored?", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"][0]["n"] == 1
    assert body["chunk_count"] == 3
    assert set(body["timings_ms"]) >= {"retrieval", "llm", "total"}
    assert body["usage"]["total_tokens"] == 132


def test_reingest_via_api_is_idempotent(client, small_corpus):
    before = client.get("/health").json()["index"]["vectors"]
    report = client.post("/ingest", json={"path": str(small_corpus)}).json()
    assert report["unchanged"] == 4 and report["vectors_after"] == before


def test_bad_requests(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "filters": {"colour": "red"}}).status_code == 400
    assert client.post("/ingest", json={"path": "/does/not/exist"}).status_code == 404
