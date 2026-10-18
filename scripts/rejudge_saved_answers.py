"""Rejudge faithfulness for all saved answered questions with source metadata.

Answers and passage contents are frozen; no new answers are generated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docqa.evaluation.controlled import restore_baseline_cases
from docqa.evaluation.experiments import AuditedClient, experiment_config, save_json, usage_summary
from docqa.evaluation.judge import Judge
from docqa.passages import coerce_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/answers.jsonl")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="results/review-v2/faithfulness-all")
    parser.add_argument("--prompt-for-key", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("Choose a fresh output directory")
    baseline = [json.loads(line) for line in Path(args.baseline).read_text(encoding="utf8").splitlines()]
    ids = [r["id"] for r in baseline if r.get("judge_faithfulness") and not r["abstained"]]
    cases = restore_baseline_cases(Path(args.baseline), Path(args.corpus), ids)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "input_cases.json", cases)
    config = experiment_config("JUDGE", "claude-sonnet-5-5")
    if args.prompt_for_key:
        config.api_key = getpass.getpass("Judge API key (hidden): ")
    random.Random(20261018).shuffle(cases)
    rows = []
    for index, case in enumerate(cases):
        client = AuditedClient(config, out / "audit.jsonl", {"role": "judge", "case_id": case["id"]})
        result = Judge(client).faithfulness(case["question"], case["answer"], coerce_passages(case["passages"]))
        rows.append({"id": case["id"], "frozen_input_sha256": case["frozen_input_sha256"],
                     "baseline_score": case["baseline_faithfulness"], "judgement": asdict(result)})
        save_json(out / "trials.json", rows)
        print(f"faithfulness {index + 1}/{len(cases)} {case['id']}: {result.score}", flush=True)
    scores = [r["judgement"]["score"] for r in rows if r["judgement"]["score"] is not None]
    report = {"kind": "LIVE_FROZEN_ANSWER_FAITHFULNESS", "status": "complete" if len(scores) == len(cases) else "incomplete",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cases": len(cases), "valid": len(scores),
              "mean_faithfulness": sum(scores) / len(scores) if scores else None,
              "historical_mean_on_same_answers": sum(r["baseline_score"] for r in rows) / len(rows),
              "usage": usage_summary(out / "audit.jsonl"),
              "scope": "Faithfulness only: all saved generated answers and exact passage texts frozen. Titles, paths, sections and pages are reconstructed with verified chunk content IDs. Not a new end-to-end generation run; relevance and correctness are not rerun."}
    save_json(out / "report.json", report)
    (out / "RESULTS.md").write_text(
        f"# Source-inclusive faithfulness replay\n\n{report['valid']}/{report['cases']} valid judgments. "
        f"Mean: {report['mean_faithfulness']}; historical mean: {report['historical_mean_on_same_answers']}.\n\n"
        + report["scope"] + "\n", encoding="utf8")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
