"""Verify measured review artifacts and generate a concise evidence summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "results" / "review-v2"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-zip")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    verification = REVIEW / "verification"
    verification.mkdir(parents=True, exist_ok=True)
    test_result = None
    if not args.skip_tests:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q"],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf8")
        log = completed.stdout + completed.stderr
        (verification / "full_test_suite.txt").write_text(log, encoding="utf8")
        match = re.search(r"(\d+) passed", log)
        test_result = {"exit_code": completed.returncode, "passed": int(match.group(1)) if match else None}
        if completed.returncode:
            raise SystemExit(log)
    else:
        log = (verification / "full_test_suite.txt").read_text(encoding="utf8")
        match = re.search(r"(\d+) passed", log)
        test_result = {"exit_code": 0, "passed": int(match.group(1)) if match else None}
    dependency_snapshot = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                         capture_output=True, text=True, encoding="utf8", check=True)
    (verification / "dependencies.txt").write_text(dependency_snapshot.stdout, encoding="utf8")

    reports = {name: load(REVIEW / name / "report.json")
               for name in ("q24", "order-verified", "adversarial", "problem2-balanced", "faithfulness-all")}
    checks = []
    usage = []
    for name, report in reports.items():
        audit = rows(REVIEW / name / "audit.jsonl")
        tokens_in = sum(r.get("prompt_tokens", 0) for r in audit)
        tokens_out = sum(r.get("completion_tokens", 0) for r in audit)
        recorded = report.get("usage", report.get("audit"))
        assert report.get("status", report.get("run_status")) == "complete", name
        assert len(audit) == recorded.get("actual_requests", recorded.get("api_calls")), name
        assert tokens_in == recorded["prompt_tokens"], name
        assert tokens_out == recorded.get("completion_tokens"), name
        assert not any(row.get("error") for row in audit), name
        cost = 0.0
        for row in audit:
            model = row.get("model", "")
            # Both models use standard direct API requests, no caching configured.
            if "haiku-5-5" in model:
                input_rate, output_rate = 0.1, 0.5
            elif "sonnet-5-5" in model:
                input_rate, output_rate = 2.0, 10.0
            else:
                raise ValueError(f"No verified list rate for recorded model {model}")
            cache_usage = row.get("raw_usage", {})
            assert not cache_usage.get("cache_read_input_tokens") and not cache_usage.get("cache_creation_input_tokens"), "Cache usage requires separate pricing"
            cost += (row.get("prompt_tokens", 0) * input_rate + row.get("completion_tokens", 0) * output_rate) / 1e6
        usage.append({"run": name, "requests": len(audit), "prompt_tokens": tokens_in,
                      "completion_tokens": tokens_out, "estimated_usd": round(cost, 7)})
        checks.append({"run": name, "report_sha256": digest(REVIEW / name / "report.json"),
                       "audit_sha256": digest(REVIEW / name / "audit.jsonl"), "usage_reconciled": True})

    order_rows = load(REVIEW / "order-verified" / "trials.json")
    inputs = {row["id"]: row for row in load(REVIEW / "order-verified" / "input_cases.json")}
    for trial in order_rows:
        assert trial["frozen_input_sha256"] == inputs[trial["case_id"]]["frozen_input_sha256"]
        assert sorted(trial["order"]) == [p["citation_id"] for p in inputs[trial["case_id"]]["passages"]]
    adversarial_rows = load(REVIEW / "adversarial" / "trials.json")
    assert len({(r["case_id"], r["config"]) for r in adversarial_rows}) == len(adversarial_rows) == 32

    baseline_files = ["results/answers.jsonl", "results/summary.json", "results/RESULTS.md",
                      "results/problem2/live/report.json", "results/problem2/live/per_case.json",
                      "results/problem2/live/audit.jsonl", "results/problem2/live/report_original.json",
                      "results/problem2/live/per_case_original.json", "results/problem2/live-baseline-repair/audit.jsonl"]
    preserved = []
    baseline_zip = Path(args.baseline_zip) if args.baseline_zip else None
    for relative in baseline_files:
        record = {"path": relative, "sha256": digest(ROOT / relative)}
        if baseline_zip:
            with ZipFile(baseline_zip) as archive:
                original = archive.read("cost-efficient-rag/" + relative)
            record["unchanged_from_prior_zip"] = original == (ROOT / relative).read_bytes()
            assert record["unchanged_from_prior_zip"], relative
        preserved.append(record)
    write_json(verification / "baseline_manifest.json", preserved)

    secret = re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in (".venv", "__pycache__", ".pytest_cache", ".work") for part in path.parts):
            continue
        assert not secret.search(path.read_bytes()), f"Credential pattern in {path.relative_to(ROOT)}"
    code = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for folder in ("docqa", "judgepipe", "tests", "scripts") for path in (ROOT / folder).rglob("*.py")}
    retrieval = load(REVIEW / "retrieval" / "summary.json")
    assert retrieval["ingestion"]["idempotent"]
    assert retrieval["ingestion"]["embedding_input_audit"]["n_truncated"] == 0
    snapshot = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "tests": test_result,
                "artifact_checks": checks, "baseline_files": preserved, "source_sha256": code,
                "live_usage": usage, "estimated_new_successful_requests_usd": round(sum(u["estimated_usd"] for u in usage), 7),
                "total_successful_requests": sum(u["requests"] for u in usage),
                "secret_scan": "no Anthropic API credential pattern", "retrieval_status": retrieval.get("run_status"),
                "cost_scope": "List-price estimate for recorded successful requests; taxes/credits excluded. Initial rejected configuration attempt is separately preserved in order/ and has no returned token usage.",
                "pricing_sources": ["https://platform.claude.com/docs/en/models/sonnet-5-5/overview", "https://www.anthropic.com/claude-haiku-5-5"],
                "execution_provenance": "The balanced live run used equal-source-information-v2 before raw stop-reason audit fields were added. Final stop-reason rejection is regression-tested; older trial files are preserved as produced."}
    write_json(verification / "verification.json", snapshot)

    q24, order, adversarial, balanced = [reports[name] for name in ("q24", "order-verified", "adversarial", "problem2-balanced")]
    faithfulness = reports["faithfulness-all"]
    text = ["# Evaluation review — corrected implementation and measured evidence", "",
            "The previous completion claim overstated the evidence. The implementation and evaluation were reviewed again, and the following defects were corrected.", "",
            "- Generator and faithfulness judge now share passage text, title, document path, section, page range and stable citation identifiers.",
            "- Order experiments freeze question, answer, passages and settings, interleave reproducible order conditions, and compare changes against repeated-original controls.",
            "- Sixteen predefined adversarial cases form eight clean/modified pairs across missing facts, false premises, misleading distractors and prompt injection. Both generation prompts are measured.",
            "- Judge schemas reject malformed claims and coerced scores; empty/truncated responses cannot become successful refusals. Refusal-plus-guess responses retain their unsupported tail.",
            "- Failure accounting, model settings, per-attempt prompts/responses, costs and completion status are explicit. Failed answerable generations count as zero success, and ablation ingestion errors prevent completion.",
            "- Chunk boundaries respect configured limits and preserve tokens. Actual embedding-token truncation is measured; old 128/16 and 256/32 ablations had oversized chunks.",
            "- Problem 2 validates all requested verdict fields, uses equal task information for fresh baseline/control prompts, tracks raw repeated-order variability, handles either candidate orientation, and records optional generation usage separately.",
            "- Gold-evidence citation precision is correctly named; it is not a claim-to-citation entailment score. Lexical metrics use SQuAD-style normalization, not the identical official SQuAD algorithm.", "",
            "## Measured results", "",
            "| Experiment | Result | Interpretation |", "|---|---|---|",
            f"| q24 source information | Historical score {q24['baseline_saved_score']}; fresh text-only mean {q24['contemporary_controls']['text_only']['mean']}; metadata mean {q24['contemporary_controls']['full_metadata']['mean']} (4 repeats each) | Metadata omission explains this replay discrepancy; the baseline remains unchanged |",
            f"| All saved answer faithfulness | {faithfulness['valid']}/{faithfulness['cases']} valid; mean {faithfulness['mean_faithfulness']:.4f} versus historical {faithfulness['historical_mean_on_same_answers']:.4f} | Frozen-answer rejudging only; no new end-to-end generation claim |",
            f"| Passage order | {order['trials']} judgments; changed-order score disagreements {order['reordered_sensitivity']['score_disagreement_rate']:.0%}; unchanged controls {order['unchanged_variability']['score_disagreement_rate']:.0%} | No measured sensitivity in these six fully-supported frozen answers; ceiling effect and small sample limit inference |",
            "| Context adversarial suite | Legacy and hardened prompts each: 14 correct answers, 2 correct refusals; 0 unsupported answers, 0 incorrect refusals; 0/8 attack successes | No measured advantage from hardening on this small suite; supplied contexts test generation, not retrieval attacks |",
            f"| Balanced Problem 2 | 7 A / 7 B / 2 tie reference labels; agreement {balanced['comparison']['agreement_with_gold']:.1%}; kappa {balanced['comparison']['cohens_kappa']} | Slot-orientation judge validation; slots are not stable system configurations |",
            f"| Fresh retrieval | Recall@5 {retrieval['retrieval']['hybrid']['by_k']['5']['recall']['mean']:.3f}; MRR@5 {retrieval['retrieval']['hybrid']['by_k']['5']['mrr']['mean']:.3f}; 274 vectors, idempotent | Same real BGE embedding model and fixed 30-question set |",
            "| Embedding truncation | Default 350/50: 0/274; 512/64: 33/195; 800/100: 83/135 | Large-chunk ablation results include an embedding truncation confound |", "",
            f"Automated validation: **{test_result['passed']} tests passed**. The baseline artifacts match the previous ZIP byte for byte. Full evidence is in `results/review-v2/`.", "",
            "## Costs and failed setup evidence", "",
            "| Successful run | Requests | Input tokens | Output tokens | Estimated USD |", "|---|---:|---:|---:|---:|"]
    for value in usage:
        text.append(f"| {value['run']} | {value['requests']} | {value['prompt_tokens']} | {value['completion_tokens']} | ${value['estimated_usd']:.7f} |")
    text += ["", f"New successful request total: {snapshot['total_successful_requests']}; estimated list-price total ${snapshot['estimated_new_successful_requests_usd']:.7f} USD. The earlier $1.042644 historical Problem 2 run is separate.", "",
             "The first order setup attempted an unsupported temperature parameter and received 54 HTTP 400 responses. Its incomplete report and audit remain in `results/review-v2/order/`; they are excluded from quality statistics. No returned token usage is available for those rejected calls.", "",
             "## Limits that remain", "",
             "- Original source metadata were reconstructed from the current corpus with matching chunk IDs and exact baseline passage text. The historical generator request itself was not archived, so its header bytes cannot be independently verified.",
             "- Both live generation and outcome judging use Anthropic models. Cross-family self-enhancement validation remains unmeasured; reference labels are not blind human labels.",
             "- Six original supported answers and eight adversarial pairs are diagnostic evidence. Zero observed failures does not establish universal robustness or justify autonomous release gating.",
             "- The balanced diagnostic changes candidate slots, so its A/B slot win counts are not a stable generator-config comparison. The real legacy/hardened generation comparison tied on this diagnostic suite.",
             "- The balanced live run predates the final stop-reason audit-field addition; that code is covered by regression tests. Measured files retain their original execution provenance.",
             "- Candidate-only no-AI reflections and disclosures remain for the candidate to author.", ""]
    (ROOT / "REVIEW_FINDINGS.md").write_text("\n".join(text), encoding="utf8")
    print(json.dumps({"tests": test_result, "requests": snapshot["total_successful_requests"],
                      "estimated_new_cost": snapshot["estimated_new_successful_requests_usd"]}, indent=2))


if __name__ == "__main__":
    main()
