"""Consolidate a primary live run with a targeted baseline repair.

The primary run remains the source for every controlled judgment.  The repair
run supplies only baseline A/B fields that were still schema-invalid.  Legacy
root-level baseline pointwise responses are reparsed with the same strict 0-4
validation; scores are never coerced or invented.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgepipe.parsing import parse_pointwise
from judgepipe.runner import aggregate


def load(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def audit_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def unresolved(rows: list[dict]) -> int:
    missing = 0
    for row in rows:
        for key in ("pointwise_a", "pointwise_b", "baseline_pointwise", "baseline", "baseline_reverse"):
            if row.get(key, {}).get("parsed") is None:
                missing += 1
        for key, value in row.items():
            if key.startswith("pairwise_repeat_") and isinstance(value, dict):
                missing += int(value.get("AB", {}).get("parsed") is None)
                missing += int(value.get("BA", {}).get("parsed") is None)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--supplement", required=True)
    args = parser.parse_args()

    primary = Path(args.primary)
    supplement = Path(args.supplement)
    original_report = primary / "report_original.json"
    original_cases = primary / "per_case_original.json"
    current_report = load(primary / "report.json")
    if original_report.exists():
        saved_report = load(original_report)
        prior_consolidation = current_report.get("consolidation", {})
        same_consolidated_source = bool(saved_report.get("generated_at_utc")) and (
            prior_consolidation.get("primary_generated_at_utc") == saved_report["generated_at_utc"])
        if current_report != saved_report and not same_consolidated_source:
            raise ValueError("Saved original snapshots belong to a different run. Use a fresh output directory; do not combine stale originals with a new live run.")
    if not original_report.exists():
        shutil.copy2(primary / "report.json", original_report)
    if not original_cases.exists():
        shutil.copy2(primary / "per_case.json", original_cases)

    source_report = load(original_report)
    rows = load(original_cases)
    supplement_report = load(supplement / "report.json")
    if any(report.get("report_kind") != "LIVE_LLM_JUDGE" for report in (source_report, supplement_report)):
        raise ValueError("Consolidation requires actual live primary and supplement reports")
    if source_report.get("judge_model") != supplement_report.get("judge_model"):
        raise ValueError("Primary and supplement judge models differ")
    repair_rows = {row["id"]: row for row in load(supplement / "per_case.json")}

    recovered_pointwise = []
    for row in rows:
        baseline_pointwise = row.get("baseline_pointwise", {})
        if baseline_pointwise.get("parsed") is None and baseline_pointwise.get("raw"):
            parsed = parse_pointwise(baseline_pointwise["raw"])
            if parsed is not None:
                baseline_pointwise["parsed"] = parsed
                baseline_pointwise["error"] = None
                baseline_pointwise["recovered_from_strict_legacy_parse"] = True
                recovered_pointwise.append(row["id"])

        if row.get("baseline", {}).get("parsed") is None:
            repair = repair_rows.get(row["id"])
            if not repair:
                continue
            for key in ("baseline", "baseline_reverse", "baseline_winner",
                        "baseline_reverse_winner", "baseline_position_flip"):
                row[key] = repair[key]
            row["baseline_repaired_from_supplement"] = str(supplement.as_posix())

    report = aggregate(
        rows,
        mode="live",
        repeats=int(source_report["repeats"]),
        judge_model=source_report["judge_model"],
        generator_model=source_report["generator_model"],
    )
    audits = audit_rows(primary / "audit.jsonl") + audit_rows(supplement / "audit.jsonl")
    unresolved_count = unresolved(rows)
    input_rate = 2.0
    output_rate = 10.0
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in audits)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in audits)
    estimated_cost = round(
        (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000,
        6,
    )
    report["audit"] = {
        "api_calls": len(audits),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "failed_calls": sum(bool(row.get("error")) for row in audits),
        "schema_retry_calls": sum(int(row.get("attempt") or 1) > 1 for row in audits),
        "unresolved_invalid_verdicts": unresolved_count,
        "invalid_verdicts": unresolved_count,
        "generation_calls": 0,
        "estimated_judge_usd": estimated_cost,
        "pricing_assumptions_usd_per_million_tokens": {
            "input": input_rate,
            "output": output_rate,
            "currency": "USD",
            "source": "https://platform.claude.com/docs/en/models/sonnet-5-5/overview",
            "checked_on": "2026-10-18",
            "calculation": f"({prompt_tokens} * {input_rate:g} + {completion_tokens} * {output_rate:g}) / 1000000 = {estimated_cost:.6f}",
            "note": "List-price estimate from recorded input/output tokens; excludes taxes, credits, and any unrecorded cache adjustments.",
        },
        "source_audits": ["audit.jsonl", Path(os.path.relpath(supplement / "audit.jsonl", primary)).as_posix()],
    }
    expected_pairs = sum(bool(row.get("pairwise")) for row in rows)
    report["run_status"] = (
        "complete"
        if report["pointwise"]["graded"] == len(rows)
        and report["comparison"]["decisive_or_tie"] == expected_pairs
        and report["bias"]["baseline_order_swaps_tested"] == expected_pairs
        and unresolved_count == 0
        and report["audit"]["failed_calls"] == 0
        else "incomplete"
    )
    report["model_family_separation"] = source_report.get("model_family_separation", {})
    # Historical data remain attributable to the prompts that actually produced them.
    # Running this script with updated code does not rerun or upgrade old judgments.
    report["prompt_protocol_version"] = source_report.get("prompt_protocol_version", "legacy-v1")
    report["historical_prompt_limitations"] = (
        ["Legacy baseline pointwise prompts omitted source context, generator system instructions and additional criteria. Baseline/control score spread is confounded by task information and does not isolate calibration."]
        if report["prompt_protocol_version"] == "legacy-v1" else []
    )
    report["consolidation"] = {
        "primary_generated_at_utc": source_report.get("generated_at_utc"),
        "supplement_generated_at_utc": supplement_report.get("generated_at_utc"),
        "primary_prompt_protocol_version": report["prompt_protocol_version"],
        "supplement_prompt_protocol_version": supplement_report.get("prompt_protocol_version", "legacy-v1-baseline-schema-repair"),
        "recovered_legacy_baseline_pointwise_cases": recovered_pointwise,
        "baseline_pair_repair_cases": [row["id"] for row in rows if row.get("baseline_repaired_from_supplement")],
        "method": "Strictly reparse saved raw 0-4 responses and replace only missing baseline fields from the targeted live supplement.",
        "original_files": ["report_original.json", "per_case_original.json"],
    }

    (primary / "per_case.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
    (primary / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({
        "run_status": report["run_status"],
        "cases": len(rows),
        "baseline_order_swaps_tested": report["bias"]["baseline_order_swaps_tested"],
        "recovered_pointwise": len(recovered_pointwise),
        "repaired_pairs": report["consolidation"]["baseline_pair_repair_cases"],
        "unresolved": unresolved_count,
    }, indent=2))


if __name__ == "__main__":
    main()
