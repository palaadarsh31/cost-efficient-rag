# Problem 2 — Historical verification and review update (18 October 2026)

The results below describe the original live fixture run. The reviewed code and fresh equal-source, balanced-orientation validation are documented in `REVIEW_FINDINGS.md` and `results/review-v2/problem2-balanced/`. Do not treat the historical score-spread comparison as an isolated calibration experiment: its baseline omitted task information. The original 14-A/2-tie labels also provided weak protection against an A-preferring shortcut.

## Completed and verified

- The full repository dependency set installed in an isolated Python 3.12 environment, including `sqlite-vec` on Windows.
- The original repository test suite passed 50 tests. The expanded review suite and its current pass count are recorded in `results/review-v2/verification/full_test_suite.txt`.
- `python -m judgepipe validate` accepts all **16** paired cases, including **8** adversarial probes.
- DNS, TCP/TLS and HTTPS connectivity to `api.anthropic.com` pass from this computer.
- A genuine live run using `claude-sonnet-5-5` completed for all 16 cases. The consolidated report includes **173 actual judge requests**, **134,942 input tokens**, **77,276 output tokens**, **0 API failures** and **0 unresolved verdicts**.
- All 16 baseline and controlled order-swap comparisons are represented. Saved legacy-shaped baseline responses were strictly reparsed without changing scores, and the one missing baseline pair was rerun live in a targeted supplement. Original pre-consolidation reports remain alongside the final report.
- Submission tables and four PNG evidence images were generated from the measured JSON files. Their SHA-256 hashes are recorded in `results/problem2/live/evidence/manifest.json`.

## Main measured results

- Reference-label agreement: **100%**; Cohen's kappa: **1.00**.
- Test–retest decision agreement: **100%** across 16 repeated pairs.
- Baseline position-order disagreement: **6.25%**; controlled pre-consensus disagreement: **0%**.
- Wrong padded-answer wins: **0%** across 3 probes.
- Wrong confidently-written-answer wins: **0%** across 5 probes.
- A/B outcome: **A won 14**, **B won 0**, **2 ties**; declared winner A.

## Remaining limitations and user-authored work

- The recorded 134,942 input and 77,276 output tokens cost an estimated **$1.042644 USD** at Anthropic's published Claude Sonnet 5.5 list rates ($2/M input, $10/M output; checked 18 October 2026). This excludes taxes, credits and any unrecorded cache adjustments.
- The fixture answers are reference-written and their generator family is unknown, so judge/generator family separation and self-enhancement bias are **not empirically verified**.
- Reference labels are curated from supplied references rather than independently blind human labels; broader external validation remains out of scope.
- The official employer template includes candidate details, measured tables, evidence images, and the three Problem 2 reflection answers. Those responses are AI-assisted and are accompanied by a disclosure in the same section.
- The three Problem 1 personal-reflection answers supplied by the candidate have also been inserted. All six reflection fields are complete; the original assignment instructions are preserved in `submission/source_material/`.
- `audit.jsonl` contains complete prompts and judge responses. Review it before sharing outside the submission context.

## Evidence locations

- `results/problem2/live/report.json` — consolidated live metrics.
- `results/problem2/live/per_case.json` — consolidated per-case verdicts.
- `results/problem2/live/submission_tables.md` — ready-to-copy measured tables.
- `results/problem2/live/evidence/` — summary, prompt/response, order-swap and adversarial PNG evidence.
- `results/problem2/live/report_original.json` and `per_case_original.json` — untouched primary-run outputs.
- `results/problem2/live-baseline-repair/` — targeted live repair run.
- `results/problem2/verification/` — connectivity, suite validation and full test-suite logs.
