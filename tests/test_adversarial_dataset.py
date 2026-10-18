"""Validate the predefined experiment, not any claim of model robustness."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "eval" / "adversarial_context.json"
CATEGORIES = {
    "related_context_missing_fact", "false_premise",
    "misleading_distractor", "prompt_injection",
}
SOURCE_FIELDS = {
    "citation_id", "chunk_id", "doc_id", "title", "section",
    "page_start", "page_end", "text", "origin",
}


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_all_cases_are_unique_complete_and_paired(dataset):
    cases = dataset["cases"]
    assert len(cases) >= 16
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["category"] for case in cases} == CATEGORIES
    groups = defaultdict(list)
    for case in cases:
        groups[case["pair_id"]].append(case)
        assert case["question"].strip()
        assert case["expected_answer"].strip()
        assert case["success_criteria"].strip()
        assert case["expected_behavior"] in {"answer", "abstain", "correct_premise"}
        assert isinstance(case["allowed_factual_claims"], list)
        assert isinstance(case["forbidden_factual_claims"], list)
        assert case["answerable"] == (case["expected_behavior"] != "abstain")
        if case["variant"] == "modified":
            assert case["attack_target"]
        else:
            assert case["attack_target"] is None
    assert len(groups) >= 8
    for pair in groups.values():
        assert len(pair) == 2
        assert {case["variant"] for case in pair} == {"clean", "modified"}
        assert len({case["category"] for case in pair}) == 1
    assert Counter(case["category"] for case in cases) == {category: 4 for category in CATEGORIES}


def test_expectations_include_refusal_and_answer_controls(dataset):
    cases = dataset["cases"]
    clean = [case for case in cases if case["variant"] == "clean"]
    assert all(case["expected_behavior"] == "answer" for case in clean)
    assert all(case["answerable"] for case in clean)
    abstentions = [case for case in cases if not case["answerable"]]
    assert len(abstentions) >= 2
    assert all(case["category"] == "related_context_missing_fact" for case in abstentions)
    assert all(case["expected_answer"] == "I don't know based on the provided documents." for case in abstentions)
    corrected = [case for case in cases if case["expected_behavior"] == "correct_premise"]
    assert len(corrected) >= 2
    assert all(case["category"] == "false_premise" for case in corrected)


def test_context_metadata_and_citations_are_stable(dataset):
    sources = dataset["source_records"]
    assert len({source["chunk_id"] for source in sources.values()}) == len(sources)
    for source in sources.values():
        assert SOURCE_FIELDS <= source.keys()
        assert isinstance(source["citation_id"], int) and source["citation_id"] > 0
        assert all(source[field] for field in ("chunk_id", "doc_id", "title", "section", "text"))
        assert source["page_start"] is None and source["page_end"] is None
    for case in dataset["cases"]:
        ids = case["context_ids"]
        assert ids and len(set(ids)) == len(ids)
        assert set(ids) <= sources.keys()
        citations = [sources[source_id]["citation_id"] for source_id in ids]
        assert len(set(citations)) == len(citations)
    groups = defaultdict(dict)
    for case in dataset["cases"]:
        groups[case["pair_id"]][case["variant"]] = case
    for pair in groups.values():
        clean, modified = pair["clean"], pair["modified"]
        if clean["category"] == "false_premise":
            assert clean["context_ids"] == modified["context_ids"]
            assert clean["question"] != modified["question"]
        else:
            assert clean["question"] == modified["question"]
        if clean["category"] in {"misleading_distractor", "prompt_injection"}:
            # Adding a passage must not renumber the retained clean source.
            assert set(clean["context_ids"]) < set(modified["context_ids"])
            assert clean["expected_answer"] == modified["expected_answer"]


def test_clean_source_excerpts_are_verbatim_corpus_evidence(dataset):
    sources = dataset["source_records"]
    for source in sources.values():
        if source["origin"] == "corpus_excerpt":
            path = ROOT / "data" / "corpus" / source["doc_id"]
            assert path.is_file(), source["doc_id"]
            assert source["text"] in path.read_text(encoding="utf-8"), source["chunk_id"]
        else:
            assert source["origin"] in {"synthetic_distractor", "synthetic_injection"}
            assert source["doc_id"].startswith("synthetic/")
    clean_ids = {
        source_id for case in dataset["cases"] if case["variant"] == "clean"
        for source_id in case["context_ids"]
    }
    assert all(sources[source_id]["origin"] == "corpus_excerpt" for source_id in clean_ids)


def test_attacks_are_predefined_and_limitations_are_explicit(dataset):
    sources = dataset["source_records"]
    for case in dataset["cases"]:
        if case["category"] == "prompt_injection" and case["variant"] == "modified":
            injected = [sources[source_id] for source_id in case["context_ids"]
                        if sources[source_id]["origin"] == "synthetic_injection"]
            assert len(injected) == 1
            assert case["attack_target"] in injected[0]["text"]
            assert case["attack_target"] in case["forbidden_factual_claims"]
    limitations = " ".join(dataset["limitations"]).lower()
    assert "do not measure retrieval attack success" in limitations
    assert "before model execution" in limitations
