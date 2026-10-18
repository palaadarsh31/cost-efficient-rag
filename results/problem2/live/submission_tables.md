# Problem 2 — measured submission results

**Judge:** `claude-sonnet-5-5`; configured generator: `reference-written`. 
**Cases evaluated:** 16; **paired:** 16. Labels in this suite are reference-derived rather than independently blind human-scored.

## 2.3.1 Judging mode & rubric

Pointwise anchored scoring + pairwise A/B comparisons in both orders; 0–4 scores.

| Criterion | Definition / anchors | Weight |
|---|---|---:|
| Correctness | Factual accuracy and agreement with reference, when provided. 0 contradictory or fabricated; 1 mostly wrong; 2 material errors; 3 minor error; 4 fully accurate | 30% |
| Faithfulness | Claims traceable to supplied context, if any; do not reward outside knowledge. 0 unsupported; 1 mostly unsupported; 2 mixed; 3 mostly grounded; 4 fully grounded | 25% |
| Completeness | Covers the requested essentials without unrelated filler. 0 misses task; 1 large omissions; 2 partial; 3 nearly complete; 4 complete and concise | 20% |
| Instruction Following | Follows the explicit system prompt and format constraints. 0 ignores instructions; 1 serious violations; 2 some violations; 3 minor issue; 4 follows fully | 15% |
| Tone Safety | Clear, neutral tone with appropriate restraint; do not reward flattery. 0 unsafe/inappropriate; 1 severe issue; 2 noticeable issue; 3 acceptable; 4 clear and appropriate | 10% |

## 2.3.2 Bias handling

| Bias | Code mitigation | Measured baseline → controlled (if available) |
|---|---|---|
| Position | Both A/B orders, remap and tie on disagreement | 6.2% → 0.0%; post-policy position-sensitive decisive outputs = 0 (by policy, not judge debiasing) |
| Verbosity | Penalize unsupported length; padded-answer probes | Wrong padded-answer wins 0.0% → 0.0%; 3 probes |
| Self-enhancement | Independent judge/generator settings; require known different source families | Not empirically measured for this prewritten fixture suite |
| Sycophancy/style | Specific per-criterion rationales; confidently wrong probes | Wrong confident-answer wins 0.0% → 0.0%; 5 probes |
| Score clustering | Anchored 0/2/4 rubric; track standard deviation | Pointwise score std dev 0.14 → 0.37 |

## 2.3.3 Judge validation

| Measurement | Result | Notes |
|---|---|---|
| Agreement with reference labels | 100.0% | 16 paired cases |
| Cohen's kappa | 1.00 | Reference-derived labels |
| Test–retest agreement | 100.0% | 16 repeated pairs |
| Position flip before order consensus | 0.0% | 16 swapped trials |
| Wrong padded answer chosen | 0.0% | 3 probes |
| Wrong confident answer chosen | 0.0% | 5 probes |

## 2.3.4 A/B comparison & winner

| Config | Pointwise pass rate (>= 3) | Mean score (0–4) | Pairwise wins (consensus) |
|---|---:|---:|---:|
| A | 93.8% | 3.81 | 14 |
| B | 12.5% | 1.32 | 0 |

**Declared winner:** A (ties: 2).

## Tokens and audit trail

Judge requests: 173; input tokens: 134942; output tokens: 77276; schema retry calls: 20; unresolved invalid verdicts: 0.
Estimated judge cost at the recorded list rates: $1.042644 USD ($2/M input; $10/M output).

Actual evidence: `report.json`, `per_case.json`, and `audit.jsonl` in this run folder.

**Candidate-only fields:** all screenshots are supplied in `evidence/`. Write all explicitly no-AI reflection answers yourself. Ensure usage disclosures are accurate.
