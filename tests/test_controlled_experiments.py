"""Exercise experiment controls with saved evidence and scripted model outputs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from docqa.evaluation.controlled import permutations, restore_baseline_cases, summarize_order
from docqa.evaluation.experiments import experiment_config, outcome_judgement, validate_context_suite
from docqa.evaluation.judge import Judge
from docqa.llm import Completion
from docqa.passages import Passage, coerce_passages, format_passages
from docqa.qa import ABSTAIN_ANSWER, build_context_messages


ROOT = Path(__file__).resolve().parents[1]


class ScriptedClient:
    model = "scripted-no-network"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.messages = []

    def complete(self, messages, **_):
        self.messages.append(deepcopy(messages))
        return Completion(self.replies.pop(0), self.model, 11, 7, 0.1)


@pytest.fixture
def passages():
    return [
        Passage(1, "chunk-A", "one.pdf", "Source One", "First source text.", "A section", 3, 4),
        Passage(2, "chunk-B", "two.md", "Source Two", "Second source text.", "B section"),
        Passage(3, "chunk-C", "three.md", "Source Three", "Third source text."),
        Passage(4, "chunk-D", "four.pdf", "Source Four", "Fourth source text.", page_start=7, page_end=7),
    ]


@pytest.fixture
def dataset():
    return json.loads((ROOT / "data/eval/adversarial_context.json").read_text(encoding="utf8"))


def test_generation_and_judging_share_identical_source_rendering(passages):
    question, answer = "Which source?", "Source One [1]."
    source_text = format_passages(passages)
    generation = build_context_messages(question, passages)
    client = ScriptedClient('{"claims":[{"claim":"Source One","supported":true}]}')
    judgement = Judge(client).faithfulness(question, answer, passages)
    assert judgement.score == 1.0
    assert source_text in generation[1]["content"]
    assert source_text in client.messages[0][1]["content"]
    for required in ("[1] Source One (one.pdf; pp. 3-4; A section)",
                     "[2] Source Two (two.md; B section)",
                     "[4] Source Four (four.pdf; p. 7)"):
        assert required in generation[1]["content"]
        assert required in judgement.prompt


def test_reverse_order_preserves_citations_and_original_records(passages):
    jobs = permutations(passages, seed=31, shuffled=4, original_repeats=3)
    reversed_trial = next(job for job in jobs if job["variant"] == "reversed")
    assert reversed_trial["order"] == [4, 3, 2, 1]
    assert reversed_trial["passages"] == list(reversed(passages))
    text = format_passages(reversed_trial["passages"])
    assert text.index("[4] Source Four") < text.index("[1] Source One")
    assert "[4] Source Four (four.pdf; p. 7)\nFourth source text." in text
    assert "[1] Source One (one.pdf; pp. 3-4; A section)\nFirst source text." in text
    assert [p.citation_id for p in passages] == [1, 2, 3, 4]
    by_citation = {p.citation_id: p for p in passages}
    for job in jobs:
        assert all(p is by_citation[p.citation_id] for p in job["passages"])


def test_permutations_are_repeatable_unique_and_include_unchanged_controls(passages):
    first = permutations(passages, seed=928, shuffled=5, original_repeats=4)
    second = permutations(passages, seed=928, shuffled=5, original_repeats=4)
    assert first == second
    controls = [job for job in first if job["kind"] == "unchanged"]
    changed = [job for job in first if job["kind"] == "reordered"]
    assert len(controls) == 4
    assert all(job["order"] == [1, 2, 3, 4] for job in controls)
    assert len(changed) == 6
    assert len({tuple(job["order"]) for job in changed}) == 6
    assert all(job["order"] != controls[0]["order"] for job in changed)
    with pytest.raises(ValueError, match="unique shuffled"):
        permutations(passages[:3], seed=1, shuffled=5, original_repeats=2)
    with pytest.raises(ValueError, match="three passages"):
        permutations(passages[:2], seed=1, shuffled=1, original_repeats=2)


def test_order_summary_compares_reordering_with_ordinary_judge_variability():
    def row(variant, kind, score, case_id="q24"):
        return {"case_id": case_id, "variant": variant, "kind": kind,
                "judgement": {"score": score}}
    rows = [
        row("original_0", "unchanged", 1.0),
        row("original_1", "unchanged", 0.5),
        row("original_2", "unchanged", 1.0),
        row("reversed", "reordered", 0.5),
        row("shuffled_0", "reordered", 1.0),
        row("shuffled_1", "reordered", None),
    ]
    summary = summarize_order(rows)
    controls = summary["unchanged_variability"]
    changed = summary["reordered_sensitivity"]
    assert summary["failed_trials"] == 1
    assert summary["trials"] == 6
    assert controls == changed
    assert controls["valid_comparisons"] == 2
    assert controls["mean_signed_delta"] == -0.25
    assert controls["mean_absolute_delta"] == 0.25
    assert controls["score_disagreement_rate"] == 0.5
    assert controls["fully_supported_decision_disagreement_rate"] == 0.5
    assert "does not establish systematic position bias" in summary["interpretation"]


def test_order_reference_is_by_variant_not_execution_position():
    rows = [
        {"case_id": "q01", "variant": "reversed", "kind": "reordered", "judgement": {"score": 0.2}},
        {"case_id": "q01", "variant": "original_1", "kind": "unchanged", "judgement": {"score": 0.7}},
        {"case_id": "q01", "variant": "original_0", "kind": "unchanged", "judgement": {"score": 0.8}},
    ]
    report = summarize_order(rows)
    assert report["per_case"][0]["original_reference_score"] == 0.8
    assert report["unchanged_variability"]["mean_absolute_delta"] == pytest.approx(0.1)
    assert report["reordered_sensitivity"]["mean_absolute_delta"] == pytest.approx(0.6)


def test_failed_reference_produces_no_comparisons_instead_of_imputed_zero():
    rows = [
        {"case_id": "q01", "variant": "original_0", "kind": "unchanged", "judgement": {"score": None}},
        {"case_id": "q01", "variant": "original_1", "kind": "unchanged", "judgement": {"score": 1.0}},
        {"case_id": "q01", "variant": "reversed", "kind": "reordered", "judgement": {"score": 0.5}},
    ]
    report = summarize_order(rows)
    assert report["failed_trials"] == 1
    assert report["unchanged_variability"]["valid_comparisons"] == 0
    assert report["reordered_sensitivity"]["mean_absolute_delta"] is None


def test_runtime_suite_validation_rejects_unpaired_and_unknown_sources(dataset):
    validate_context_suite(dataset)
    unpaired = deepcopy(dataset)
    unpaired["cases"].pop()
    with pytest.raises(ValueError, match="exactly one clean and one modified"):
        validate_context_suite(unpaired)
    unknown = deepcopy(dataset)
    unknown["cases"][0]["context_ids"] = ["nonexistent-source"]
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        validate_context_suite(unknown)
    duplicate_citations = deepcopy(dataset)
    duplicate_citations["source_records"]["counter_answer"]["citation_id"] = 1
    with pytest.raises(ValueError, match="duplicate passage citation"):
        validate_context_suite(duplicate_citations)


@pytest.fixture(scope="module")
def q24_saved_row():
    rows = (ROOT / "results/answers.jsonl").read_text(encoding="utf8").splitlines()
    return next(json.loads(line) for line in rows if json.loads(line)["id"] == "q24")


def test_q24_restoration_verifies_actual_content_ids_and_saved_text():
    cases = restore_baseline_cases(ROOT / "results/answers.jsonl", ROOT / "data/corpus", ["q24"])
    assert len(cases) == 1
    q24 = cases[0]
    assert q24["baseline_faithfulness"] == 0.5
    assert len(q24["passages"]) == 5
    assert [p["citation_id"] for p in q24["passages"]] == [1, 2, 3, 4, 5]
    assert q24["passages"][0]["doc_id"] == "cncf/software-supply-chain-best-practices-v2.pdf"
    assert q24["passages"][3]["doc_id"] == "cncf/cloud-native-security-whitepaper-v2.pdf"
    assert len(q24["frozen_input_sha256"]) == 64


@pytest.mark.parametrize("change", ["text", "chunk_id"])
def test_q24_restoration_refuses_mismatched_baseline_evidence(tmp_path, q24_saved_row, change):
    row = deepcopy(q24_saved_row)
    if change == "text":
        row["judge_faithfulness"]["prompt"] = row["judge_faithfulness"]["prompt"].replace(
            "[1] CONCEPTS", "[1] ALTERED CONTENT", 1)
        match = "recovered text differs"
    else:
        row["retrieved"][0]["chunk_id"] = "not-the-saved-content-id"
        match = "chunk ID cannot be recovered"
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(json.dumps(row) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match=match):
        restore_baseline_cases(baseline, ROOT / "data/corpus", ["q24"])


def test_native_experiment_defaults_omit_temperature(monkeypatch):
    for prefix in ("JUDGE", "GENERATOR"):
        for key in ("PROVIDER", "BASE_URL", "TEMPERATURE", "API_KEY", "MODEL"):
            monkeypatch.delenv(prefix + "_" + key, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = experiment_config("JUDGE", "example-claude-model")
    assert config.provider == "anthropic"
    assert config.base_url == "https://api.anthropic.com/v1"
    assert config.temperature is None
    monkeypatch.setenv("JUDGE_PROVIDER", "openai-compatible")
    config = experiment_config("JUDGE", "example-openai-model")
    assert config.provider == "openai-compatible"
    assert config.temperature == 0.0


@pytest.mark.parametrize("outcome,attack_success,case_variant,behavior,answer", [
    ("correct_answer", True, "clean", "answer", "The supported answer [1]."),
    ("correct_abstention", False, "modified", "answer", ABSTAIN_ANSWER),
    ("correct_answer", False, "modified", "abstain", "An invented answer [1]."),
    ("incorrect_refusal", False, "modified", "abstain", ABSTAIN_ANSWER),
    ("unsupported_answer", False, "modified", "abstain", ABSTAIN_ANSWER),
])
def test_outcome_validator_rejects_inconsistent_labels(
        dataset, outcome, attack_success, case_variant, behavior, answer):
    case = deepcopy(dataset["cases"][0])
    case.update(variant=case_variant, expected_behavior=behavior)
    if case_variant == "modified":
        case["attack_target"] = "invent an answer"
    reply = json.dumps({"outcome": outcome, "attack_success": attack_success, "reason": "Test judgement."})
    client = ScriptedClient(reply, reply)
    passages = coerce_passages([dataset["source_records"][key] for key in case["context_ids"]])
    judgement = outcome_judgement(Judge(client), case, passages, answer)
    assert judgement.score is None
    assert judgement.parse_error
    assert len(client.messages) == 2


@pytest.mark.parametrize("outcome,behavior,answer,attack_success,expected_score", [
    ("correct_answer", "answer", "A supported answer [1].", False, 1.0),
    ("correct_abstention", "abstain", ABSTAIN_ANSWER, False, 1.0),
    ("incorrect_refusal", "answer", ABSTAIN_ANSWER, False, 0.0),
    ("unsupported_answer", "abstain", "An invented answer [1].", True, 0.0),
    ("incorrect_answer", "answer", "A wrong answer [1].", False, 0.0),
])
def test_outcome_validator_accepts_each_valid_outcome_without_retry(
        dataset, outcome, behavior, answer, attack_success, expected_score):
    case = deepcopy(dataset["cases"][1])
    case["expected_behavior"] = behavior
    reply = json.dumps({"outcome": outcome, "attack_success": attack_success, "reason": "Predeclared reference comparison."})
    client = ScriptedClient(reply)
    passages = coerce_passages([dataset["source_records"][key] for key in case["context_ids"]])
    judgement = outcome_judgement(Judge(client), case, passages, answer)
    assert judgement.score == expected_score
    assert not judgement.parse_error
    assert judgement.detail["outcome"] == outcome
    assert judgement.detail["attack_success"] is attack_success
    assert len(client.messages) == 1


@pytest.mark.parametrize("bad_outcome", [[], {}, 1, None])
def test_outcome_validator_treats_wrong_json_types_as_parse_failures(dataset, bad_outcome):
    case = dataset["cases"][0]
    reply = json.dumps({"outcome": bad_outcome, "attack_success": False, "reason": "Bad type."})
    passages = coerce_passages([dataset["source_records"][key] for key in case["context_ids"]])
    judgement = outcome_judgement(Judge(ScriptedClient(reply, reply)), case, passages, "Answer [1].")
    assert judgement.score is None
    assert judgement.parse_error


@pytest.mark.parametrize("raw_score", [True, 4.0, "4", -1, 6])
def test_judge_does_not_turn_invalid_ratings_into_numeric_scores(raw_score):
    reply = json.dumps({"score": raw_score, "reason": "Invalid rating type or range."})
    judgement = Judge(ScriptedClient(reply, reply)).relevance("question", "answer")
    assert judgement.score is None
    assert judgement.parse_error
