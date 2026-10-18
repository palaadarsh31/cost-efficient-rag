import json

from docqa.ingest import ingest_path
from docqa.qa import ABSTAIN_ANSWER, QAPipeline, build_messages, is_abstention, parse_citations
from docqa.retrieval import Retriever, rrf_fuse
from docqa.store import VectorStore

from conftest import FakeLLM


def _pipeline(small_corpus, settings, embedder, llm):
    store = VectorStore(settings.db_path)
    ingest_path(small_corpus, store, embedder, settings)
    return QAPipeline(Retriever(store, embedder, settings), llm, settings)


def test_answer_has_citations_usage_and_timings(small_corpus, settings, embedder):
    llm = FakeLLM("Store config in environment variables [1]. Also see [2][9].")
    pipe = _pipeline(small_corpus, settings, embedder, llm)
    result = pipe.answer("Where does a twelve-factor app store config?", k=3)

    assert not result.abstained
    assert [c.n for c in result.citations] == [1, 2]
    assert result.invalid_citations == [9]
    assert result.chunk_count == 3
    assert result.usage["prompt_tokens"] == 120 and result.usage["completion_tokens"] == 12
    assert result.timings_ms["llm"] == 5.0 and result.timings_ms["total"] >= result.timings_ms["retrieval"]
    # the prompt numbers the passages that were retrieved
    user_msg = llm.calls[0][-1]["content"]
    assert "[1]" in user_msg and "[3]" in user_msg and "Question:" in user_msg

    logged = [json.loads(line) for line in open(settings.query_log_path)]
    assert logged[-1]["chunk_count"] == 3 and logged[-1]["usage"]["total_tokens"] == 132


def test_similarity_gate_skips_the_llm(small_corpus, settings, embedder):
    llm = FakeLLM()
    pipe = _pipeline(small_corpus, settings.copy(min_score=0.99), embedder, llm)
    result = pipe.answer("Who won the 2018 FIFA World Cup?")
    assert result.abstained and result.abstain_reason == "no_relevant_context"
    assert result.answer == ABSTAIN_ANSWER
    assert llm.calls == []
    assert result.usage["total_tokens"] == 0


def test_model_abstention_is_recognised(small_corpus, settings, embedder):
    pipe = _pipeline(small_corpus, settings, embedder,
                     FakeLLM("I don't know based on the provided documents."))
    result = pipe.answer("What port does the Pushgateway use?")
    assert result.abstained and result.abstain_reason == "model_abstained"
    assert result.citations == []


def test_uncited_answer_is_flagged(small_corpus, settings, embedder):
    pipe = _pipeline(small_corpus, settings, embedder, FakeLLM("Environment variables."))
    result = pipe.answer("Where is config stored?")
    assert not result.abstained and result.uncited


def test_metadata_filter_restricts_results(small_corpus, settings, embedder):
    pipe = _pipeline(small_corpus, settings, embedder, FakeLLM())
    for mode in ("dense", "hybrid"):
        r = pipe.retriever.retrieve("how fast should code review be", k=5, mode=mode,
                                    filters={"collection": "twelve-factor"})
        assert r.hits and all(h.chunk.collection == "twelve-factor" for h in r.hits)
        r = pipe.retriever.retrieve("logs", k=5, mode=mode, filters={"doc_type": "html"})
        assert all(h.chunk.doc_type == "html" for h in r.hits)


def test_helpers():
    assert parse_citations("a [1] b [3][1] c [7]", 5) == ([1, 3], [7])
    assert is_abstention("I don’t know based on the provided documents.")
    assert is_abstention("i do not know")
    assert not is_abstention("Use SIGTERM [1].")
    assert rrf_fuse([[1, 2, 3], [3, 1]])[0][0] == 1
    msgs = build_messages("q?", [])
    assert msgs[0]["role"] == "system" and ABSTAIN_ANSWER in msgs[0]["content"]
