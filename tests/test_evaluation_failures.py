from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import FakeLLM
from docqa.evaluation.judge import Judge
from docqa.evaluation.labels import Question
from docqa.evaluation.run import evaluate_answers, llm_cost
from docqa.llm import Completion, LLMError
from docqa.qa import ABSTAIN_ANSWER, QAPipeline, is_abstention
from docqa.retrieval import Retrieval


def test_generation_error_is_not_a_correct_refusal():
    class FailedPipeline:
        def answer(self, question):
            raise LLMError("empty response", completion=Completion("", "m", 20, 0, 3.0))
    rows = []
    summary = evaluate_answers(FailedPipeline(), Judge(FakeLLM()),
                               [Question("missing", "An unanswerable fact?", None, False)], {}, rows)
    assert summary["generation_failures"] == 1
    assert summary["abstention"]["unanswerable_abstained"] == 0
    assert summary["abstention"]["unanswerable_total"] == 1
    assert summary["citations"]["answered"] == 0
    assert rows[0]["generation_failed"] and rows[0]["abstained"] is False
    assert summary["tokens"]["prompt_total"] == 20


def test_refusal_with_unsupported_tail_is_not_erased():
    assert is_abstention(ABSTAIN_ANSWER)
    assert is_abstention("I don’t know.")
    assert not is_abstention(ABSTAIN_ANSWER + " However, the port is 9091.")
    assert not is_abstention("The response should not say \"" + ABSTAIN_ANSWER + "\"")


def test_generation_failure_on_answerable_case_stays_in_quality_denominators():
    class FailedPipeline:
        def answer(self, question):
            raise LLMError("transport failure")
    rows = []
    summary = evaluate_answers(FailedPipeline(), Judge(FakeLLM()),
                               [Question("answerable", "A fact?", "Supported answer", True)], {}, rows)
    for metric in ("correctness", "em", "f1", "contains_gold"):
        assert summary[metric]["mean"] == 0.0
        assert summary[metric]["n"] == 1
    assert rows[0]["timings_ms"]["total"] > 0


def test_prices_user_queries_including_gate_separately_from_generation_calls(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_M", "2")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_M", "10")
    tokens = {"prompt_mean": 1000, "completion_mean": 100,
              "prompt_total": 1000, "completion_total": 100, "n_user_queries": 2}
    cost = llm_cost(tokens)
    assert cost["usd_per_1k_generation_calls"] == pytest.approx(3.0)
    assert cost["usd_per_1k_queries"] == pytest.approx(1.5)
    assert cost["total_generation_usd"] == pytest.approx(0.003)


def test_pipeline_rejects_empty_alternate_client_completion(settings):
    fake_chunk = SimpleNamespace(chunk_id="c", doc_id="d", title="T", section=None,
                                 page_start=None, page_end=None, text="Evidence")
    hit = SimpleNamespace(rank=1, chunk=fake_chunk, similarity=0.8, score=0.8)
    result = Retrieval("q", "dense", 1, [hit], 0.8,
                       {"total": 0.2, "embed": 0.1, "search": 0.1, "fetch": 0.0})
    retriever = SimpleNamespace(retrieve=lambda *a, **k: result)
    with pytest.raises(LLMError, match="cannot count as an abstention"):
        QAPipeline(retriever, FakeLLM(""), settings).answer("q")
