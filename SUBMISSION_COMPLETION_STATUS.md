# Submission completion status

**Candidate:** Pala Adarsh  
**Email:** adarsh1@iisc.ac.in  
**Submission date:** 18 October 2026

**Total time spent:** 6 hours

## Completion declaration

The earlier 100% claim overstated what the evidence established. The reviewed implementation corrects source formatting, controls passage order against repeated-original variability, adds paired adversarial cases, fixes validation/failure accounting, and preserves the original results as historical baselines. See `REVIEW_FINDINGS.md` for current measured outcomes and limits.

All six reflection fields are filled in the Word and PDF submission. The three Problem 1 responses were supplied by the candidate and inserted into the matching sections; the Problem 2 responses and AI disclosures are retained. No candidate-answer placeholders remain in the completed submission. This records document completion, with the evaluation limitations below retained.

## Verified results

- Full expanded repository suite: see the current pass count and log in `results/review-v2/verification/`.
- Problem 2 suite schema: **16 valid paired cases**, including **8 adversarial probes**.
- Live judge: `claude-sonnet-5-5`; **173 genuine API calls**, **0 API failures**, **0 unresolved verdicts**.
- Live usage: **134,942 input tokens** and **77,276 output tokens**.
- Estimated judge cost: **$1.042644 USD** at Anthropic's published Sonnet 5.5 list rates of $2/M input and $10/M output, checked 18 October 2026. Taxes, credits and unrecorded cache adjustments are excluded.
- Reference-label agreement and test–retest agreement: **100%**; Cohen's kappa: **1.00**.
- Position disagreement: **6.25% baseline → 0.00% controlled** on this suite.
- Wrong padded-answer wins: **0/3**; wrong confidently-written-answer wins: **0/5**.
- The official submission template contains candidate details, setup instructions, both architecture diagrams, Problem 1 and Problem 2 result tables, technical trade-off answers, and seven measured evidence images.
- Current source-information replay, passage-order controls, paired context-adversarial trials, real-embedding retrieval rerun, and balanced Problem 2 diagnostics are in `results/review-v2/`. The old numbers above remain historical, not rerun claims.
- Credential scan: no Anthropic API secret pattern is present in the submission files or package.

## Reflection answers included

- Problem 1: query walkthrough, initial failure and debugging explanation, and AI disclosure.
- Problem 2: position-bias explanation, adversarial-result interpretation, and AI disclosure.
- Editable text: `SUBMISSION_ANSWERS.md`; formatted copies: `submission/Pala_Adarsh_Applied_AI_ML_Submission.docx` and `.pdf`.

## Explicit limitation retained for accuracy

The Problem 2 fixtures are reference-written and do not have an attributable generator-model family. The judge and generator are independently configurable in code, but the submitted run does not claim empirical cross-model self-enhancement measurement. Reference labels are curated from supplied references, not independently blind human labels.

The context robustness experiment uses Anthropic for both generation and outcome judging. The six fixed answers in the passage-order experiment are all fully supported, creating a ceiling-limited diagnostic. The larger chunk-size ablations also include embedding truncation. These limitations remain explicit; no universal bias-free or release-safe claim is made.

## Included submission files

- `submission/Pala_Adarsh_Applied_AI_ML_Submission.docx`
- `submission/Pala_Adarsh_Applied_AI_ML_Submission.pdf`
- `results/problem2/live/report.json`
- `results/problem2/live/per_case.json`
- `results/problem2/live/audit.jsonl`
- `results/problem2/live/evidence/`

Official Claude pricing source: https://platform.claude.com/docs/en/models/sonnet-5-5/overview
