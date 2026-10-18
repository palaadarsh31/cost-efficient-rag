"""Turn an ACTUAL live judge report into assignment evidence tables.

This intentionally does not create the personal, no-AI reflection answers.
It rejects offline test-double reports to prevent misleading submissions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgepipe.rubric import RUBRIC


def fmt(value, multiplier=1):
    if value is None:
        return 'Not measured'
    return f'{value * multiplier:.2f}'


def pct(value):
    return 'Not measured' if value is None else f'{value * 100:.1f}%'


def report_markdown(data: dict) -> str:
    if data.get('report_kind') != 'LIVE_LLM_JUDGE':
        raise ValueError('Refusing to present test-double or unknown results as live LLM evidence')
    if data.get('run_status') != 'complete':
        raise ValueError('Incomplete live judge run: fix failures or report the limitations before generating submission tables')
    p, c, b, audit = (data[k] for k in ('pointwise','comparison','bias','audit'))
    lines = [
        '# Problem 2 — measured submission results',
        '',
        f"**Judge:** `{data.get('judge_model', 'unspecified')}`; configured generator: "
        f"`{data.get('generator_model', 'unspecified')}`. ",
        f"**Cases evaluated:** {data.get('cases_total', 0)}; **paired:** {data.get('cases_with_two_outputs', 0)}. "
        'Labels in this suite are reference-derived rather than independently blind human-scored.',
        'Protocol: ' + data.get('prompt_protocol_version', 'legacy-v1') + '. '
        + ('Historical baseline pointwise prompts omitted task information; score-spread changes cannot isolate calibration.'
           if data.get('prompt_protocol_version', 'legacy-v1') == 'legacy-v1'
           else 'Fresh baseline and control prompts share task evidence.'),
        'A/B wins identify stable configurations only when A/B retain the same system meaning across cases. A slot-balanced orientation diagnostic cannot declare a superior system.',
        '',
        '## 2.3.1 Judging mode & rubric',
        '',
        'Pointwise anchored scoring + pairwise A/B comparisons in both orders; 0–4 scores.',
        '',
        '| Criterion | Definition / anchors | Weight |',
        '|---|---|---:|',
    ]
    for name, entry in RUBRIC.items():
        lines.append(f"| {name.replace('_',' ').title()} | {entry['definition']} {entry['anchors']} | {entry['weight']:.0%} |")
    lines += [
        '', '## 2.3.2 Bias handling', '',
        '| Bias | Code mitigation | Measured baseline → controlled (if available) |',
        '|---|---|---|',
        '| Position | Both A/B orders, remap and tie on disagreement | '
        f"{pct(b.get('baseline_uncontrolled_position_order_disagreement_rate'))} → "
        f"{pct(b.get('position_order_disagreement_rate_BEFORE_consensus'))}; "
        'post-policy position-sensitive decisive outputs = '
        f"{b.get('position_sensitive_decisions_after_consensus', 'unknown')} (by policy, not judge debiasing) |",
        '| Verbosity | Penalize unsupported length; padded-answer probes | '
        f"Wrong padded-answer wins {pct(b.get('padded_answer_wrong_win_rate_baseline'))} → "
        f"{pct(b.get('padded_answer_wrong_win_rate'))}; "
        f"{b.get('padded_probes', 0)} probes |",
        '| Self-enhancement | Independent judge/generator settings; require known different source families | '
        'Not empirically measured for this prewritten fixture suite |',
        '| Sycophancy/style | Specific per-criterion rationales; confidently wrong probes | '
        f"Wrong confident-answer wins {pct(b.get('confidently_wrong_wrong_win_rate_baseline'))} → "
        f"{pct(b.get('confidently_wrong_wrong_win_rate'))}; "
        f"{b.get('confidently_wrong_probes', 0)} probes |",
        '| Score clustering | Anchored 0/2/4 rubric; track standard deviation | '
        f"Pointwise score std dev {fmt(b.get('score_spread_baseline_stddev_0_to_4'))} → "
        f"{fmt(b.get('score_spread_stddev_0_to_4'))} |",
        '', '## 2.3.3 Judge validation', '',
        '| Measurement | Result | Notes |',
        '|---|---|---|',
        f"| Agreement with reference labels | {pct(c.get('agreement_with_gold'))} | {c.get('gold_labeled_evaluated', 0)} paired cases |",
        f"| Cohen's kappa | {fmt(c.get('cohens_kappa'))} | Reference-derived labels |",
        f"| Test–retest agreement | {pct(b.get('test_retest_decision_agreement'))} | {b.get('test_retest_pairs', 0)} repeated pairs |",
        f"| Position flip before order consensus | {pct(b.get('position_order_disagreement_rate_BEFORE_consensus'))} | {b.get('position_reverse_trials', 0)} swapped trials |",
        f"| Wrong padded answer chosen | {pct(b.get('padded_answer_wrong_win_rate'))} | {b.get('padded_probes', 0)} probes |",
        f"| Wrong confident answer chosen | {pct(b.get('confidently_wrong_wrong_win_rate'))} | {b.get('confidently_wrong_probes', 0)} probes |",
        '', '## 2.3.4 A/B comparison & winner', '',
        '| Config | Pointwise pass rate (>= 3) | Mean score (0–4) | Pairwise wins (consensus) |',
        '|---|---:|---:|---:|',
        f"| A | {pct(p.get('pass_rate_overall_at_least_3'))} | {fmt(p.get('mean_overall_0_to_4'))} | {c.get('A_wins', 0)} |",
        f"| B | {pct(p.get('config_B_pass_rate_overall_at_least_3'))} | {fmt(p.get('config_B_mean_overall_0_to_4'))} | {c.get('B_wins', 0)} |",
        f"\n**Declared winner:** {c.get('declared_winner', 'unknown')} (ties: {c.get('ties', 0)}).",
        '', '## Tokens and audit trail', '',
        f"Judge requests: {audit.get('api_calls', 0)}; input tokens: {audit.get('prompt_tokens', 0)}; "
        f"output tokens: {audit.get('completion_tokens', 0)}; schema retry calls: "
        f"{audit.get('schema_retry_calls', 'not recorded')}; unresolved invalid verdicts: "
        f"{audit.get('unresolved_invalid_verdicts', audit.get('invalid_verdicts', 0))}.",
        f"Estimated judge cost at the recorded list rates: ${audit.get('estimated_judge_usd', 0):.6f} USD "
        f"(${audit.get('pricing_assumptions_usd_per_million_tokens', {}).get('input', 0):g}/M input; "
        f"${audit.get('pricing_assumptions_usd_per_million_tokens', {}).get('output', 0):g}/M output).",
        '', 'Actual evidence: `report.json`, `per_case.json`, and `audit.jsonl` in this run folder.',
        '', '**Candidate-only fields:** all screenshots are supplied in `evidence/`. '
        'Write all explicitly no-AI reflection answers yourself. Ensure usage disclosures are accurate.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',required=True)
    p.add_argument('--out',default=None)
    args=p.parse_args()
    inp=Path(args.report)
    out=Path(args.out) if args.out else inp.with_name('submission_tables.md')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(report_markdown(json.loads(inp.read_text(encoding='utf8'))),encoding='utf8')
    print(f'Wrote {out}')


if __name__=='__main__':
    main()
