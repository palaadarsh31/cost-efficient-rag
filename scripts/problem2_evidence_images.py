"""Render redacted evidence PNGs directly from measured Problem 2 JSON files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont


SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]+")


def safe(value: object) -> str:
    return SECRET.sub("[REDACTED]", str(value))


def font(size: int, bold: bool = False):
    choices = [
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    for choice in choices:
        if choice.exists():
            return ImageFont.truetype(str(choice), size)
    return ImageFont.load_default()


def wrap(value: object, width: int = 92) -> list[str]:
    lines: list[str] = []
    for raw in safe(value).splitlines() or [""]:
        lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False) or [""])
    return lines


def render(path: Path, title: str, subtitle: str, sections: list[tuple[str, list[str]]]) -> None:
    title_font, heading_font, body_font, small_font = font(34, True), font(24, True), font(21), font(17)
    prepared: list[tuple[str, list[str]]] = []
    line_count = 0
    for heading, values in sections:
        lines = [line for value in values for line in wrap(value)]
        prepared.append((heading, lines))
        line_count += len(lines) + 2
    width = 1500
    height = max(720, 170 + line_count * 32 + len(prepared) * 18)
    image = Image.new("RGB", (width, height), "#0b1020")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 28, width - 28, height - 28), 20, fill="#111827", outline="#334155", width=2)
    for index, colour in enumerate(("#ef4444", "#f59e0b", "#22c55e")):
        x = 62 + index * 32
        draw.ellipse((x, 54, x + 18, 72), fill=colour)
    draw.text((60, 96), title, font=title_font, fill="#f8fafc")
    draw.text((60, 142), subtitle, font=small_font, fill="#94a3b8")
    y = 188
    for heading, lines in prepared:
        draw.text((60, y), heading, font=heading_font, fill="#38bdf8")
        y += 40
        for line in lines:
            draw.text((76, y), line, font=body_font, fill="#e2e8f0")
            y += 31
        y += 18
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/problem2/live")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf8"))
    cases = json.loads((run_dir / "per_case.json").read_text(encoding="utf8"))
    case_map = {row["id"]: row for row in cases}
    audit = [json.loads(line) for line in (run_dir / "audit.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    evidence = run_dir / "evidence"

    p, c, b, a = report["pointwise"], report["comparison"], report["bias"], report["audit"]
    render(evidence / "01_live_run_summary.png", "Problem 2 — live Claude judge", "Measured from report.json; credentials are never rendered", [
        ("Run", [f"Status: {report['run_status']}   Kind: {report['report_kind']}", f"Judge: {report['judge_model']}   Cases: {report['cases_total']}/16", f"API calls: {a['api_calls']}   Failed: {a['failed_calls']}   Unresolved verdicts: {a['unresolved_invalid_verdicts']}"]),
        ("A/B result", [f"A wins: {c['A_wins']}   B wins: {c['B_wins']}   Ties: {c['ties']}", f"A mean: {p['mean_overall_0_to_4']:.2f}/4   B mean: {p['config_B_mean_overall_0_to_4']:.2f}/4", f"Agreement with labels: {c['agreement_with_gold']:.0%}   Cohen's kappa: {c['cohens_kappa']:.2f}"]),
        ("Bias checks", [f"Position disagreement baseline -> controlled: {b['baseline_uncontrolled_position_order_disagreement_rate']:.1%} -> {b['position_order_disagreement_rate_BEFORE_consensus']:.1%}", f"Test-retest agreement: {b['test_retest_decision_agreement']:.0%}", f"Wrong padded wins: {b['padded_answer_wrong_win_rate']:.1%}; wrong confident wins: {b['confidently_wrong_wrong_win_rate']:.1%}"]),
        ("Usage", [f"Input tokens: {a['prompt_tokens']:,}   Output tokens: {a['completion_tokens']:,}", f"List-price estimate: ${a['estimated_judge_usd']:.6f} USD ($2/M input, $10/M output)."]),
    ])

    prompt_row = next(row for row in audit if row.get("id") == "review_speed" and row.get("task") == "pair_AB_repeat1" and row.get("attempt") == 1)
    prompt = prompt_row["messages"][1]["content"]
    prompt_excerpt = prompt[:2200] + ("\n[excerpt ends]" if len(prompt) > 2200 else "")
    response_excerpt = prompt_row["raw_response"][:2200] + ("\n[excerpt ends]" if len(prompt_row["raw_response"]) > 2200 else "")
    render(evidence / "02_actual_prompt_response.png", "Actual judge prompt and response", "Saved API audit excerpt: review_speed / pair_AB_repeat1 / attempt 1", [
        ("Prompt excerpt", [prompt_excerpt]),
        ("Claude response excerpt", [response_excerpt]),
        ("Trace", [f"Model: {prompt_row['model']}   Input tokens: {prompt_row['prompt_tokens']}   Output tokens: {prompt_row['completion_tokens']}", f"Latency: {prompt_row['latency_ms']} ms   API error: {prompt_row['error']}"]),
    ])

    order = case_map["review_speed"]["pairwise_repeat_1"]
    render(evidence / "03_order_swap_comparison.png", "A/B versus B/A order-swap evidence", "Measured from per_case.json: review_speed, controlled repeat 1", [
        ("A/B display order", [f"Shown winner: {order['AB']['parsed']['winner']}", order['AB']['parsed']['rationale']]),
        ("B/A reversed display order", [f"Shown winner: {order['BA']['parsed']['winner']}   Canonical winner after remap: {order['BA_canonical_winner']}", order['BA']['parsed']['rationale']]),
        ("Consensus", [f"Decision after order consensus: {order['decision_after_order_consensus']}", f"Position flip: {order['position_flip']}"]),
    ])

    adversarial = case_map["wrong_confidence_1"]
    verdict = adversarial["pairwise_repeat_1"]
    render(evidence / "04_adversarial_verdict.png", "Confidently-wrong adversarial verdict", "Measured from per_case.json: wrong_confidence_1", [
        ("Probe", [f"Probe type: {adversarial['probe']}   Gold winner: {adversarial['gold_winner']}", f"Final controlled decision: {adversarial['decision']}"]),
        ("A/B judgment", [f"Shown winner: {verdict['AB']['parsed']['winner']}", verdict['AB']['parsed']['rationale']]),
        ("B/A judgment", [f"Shown winner: {verdict['BA']['parsed']['winner']}   Canonical: {verdict['BA_canonical_winner']}", verdict['BA']['parsed']['rationale']]),
        ("Aggregate", [f"Wrong confidently-written answer win rate: {b['confidently_wrong_wrong_win_rate']:.1%} across {b['confidently_wrong_probes']} probes"]),
    ])

    sources = [run_dir / "report.json", run_dir / "per_case.json", run_dir / "audit.jsonl"]
    images = sorted(evidence.glob("*.png"))
    manifest = {
        "generated_from_measured_live_results": True,
        "redaction": "Any sk-ant-* token is replaced with [REDACTED] before rendering.",
        "sources": {str(path.relative_to(run_dir)): digest(path) for path in sources},
        "images": {path.name: digest(path) for path in images},
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps({"images": [path.name for path in images], "manifest": str(evidence / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
