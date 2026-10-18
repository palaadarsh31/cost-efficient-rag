# Evaluation review — corrected implementation and measured evidence

The previous completion claim overstated the evidence. The implementation and evaluation were reviewed again, and the following defects were corrected.

- Generator and faithfulness judge now share passage text, title, document path, section, page range and stable citation identifiers.
- Order experiments freeze question, answer, passages and settings, interleave reproducible order conditions, and compare changes against repeated-original controls.
- Sixteen predefined adversarial cases form eight clean/modified pairs across missing facts, false premises, misleading distractors and prompt injection. Both generation prompts are measured.
- Judge schemas reject malformed claims and coerced scores; empty/truncated responses cannot become successful refusals. Refusal-plus-guess responses retain their unsupported tail.
- Failure accounting, model settings, per-attempt prompts/responses, costs and completion status are explicit. Failed answerable generations count as zero success, and ablation ingestion errors prevent completion.
- Chunk boundaries respect configured limits and preserve tokens. Actual embedding-token truncation is measured; old 128/16 and 256/32 ablations had oversized chunks.
- Problem 2 validates all requested verdict fields, uses equal task information for fresh baseline/control prompts, tracks raw repeated-order variability, handles either candidate orientation, and records optional generation usage separately.
- Gold-evidence citation precision is correctly named; it is not a claim-to-citation entailment score. Lexical metrics use SQuAD-style normalization, not the identical official SQuAD algorithm.

## Measured results

| Experiment | Result | Interpretation |
|---|---|---|
| q24 source information | Historical score 0.5; fresh text-only mean 0.0; metadata mean 1.0 (4 repeats each) | Metadata omission explains this replay discrepancy; the baseline remains unchanged |
| All saved answer faithfulness | 25/25 valid; mean 1.0000 versus historical 0.9800 | Frozen-answer rejudging only; no new end-to-end generation claim |
| Passage order | 54 judgments; changed-order score disagreements 0%; unchanged controls 0% | No measured sensitivity in these six fully-supported frozen answers; ceiling effect and small sample limit inference |
| Context adversarial suite | Legacy and hardened prompts each: 14 correct answers, 2 correct refusals; 0 unsupported answers, 0 incorrect refusals; 0/8 attack successes | No measured advantage from hardening on this small suite; supplied contexts test generation, not retrieval attacks |
| Balanced Problem 2 | 7 A / 7 B / 2 tie reference labels; agreement 100.0%; kappa 1.0 | Slot-orientation judge validation; slots are not stable system configurations |
| Fresh retrieval | Recall@5 1.000; MRR@5 0.868; 274 vectors, idempotent | Same real BGE embedding model and fixed 30-question set |
| Embedding truncation | Default 350/50: 0/274; 512/64: 33/195; 800/100: 83/135 | Large-chunk ablation results include an embedding truncation confound |

Automated validation: **160 tests passed**. The baseline artifacts match the previous ZIP byte for byte. Full evidence is in `results/review-v2/`.

## Costs and failed setup evidence

| Successful run | Requests | Input tokens | Output tokens | Estimated USD |
|---|---:|---:|---:|---:|
| q24 | 8 | 21364 | 1598 | $0.0587080 |
| order-verified | 54 | 153720 | 10642 | $0.4138600 |
| adversarial | 64 | 44899 | 8136 | $0.1060507 |
| problem2-balanced | 144 | 129237 | 65651 | $0.9149840 |
| faithfulness-all | 25 | 70692 | 4062 | $0.1820040 |

New successful request total: 295; estimated list-price total $1.6756067 USD. The earlier $1.042644 historical Problem 2 run is separate.

The first order setup attempted an unsupported temperature parameter and received 54 HTTP 400 responses. Its incomplete report and audit remain in `results/review-v2/order/`; they are excluded from quality statistics. No returned token usage is available for those rejected calls.

## Limits that remain

- Original source metadata were reconstructed from the current corpus with matching chunk IDs and exact baseline passage text. The historical generator request itself was not archived, so its header bytes cannot be independently verified.
- Both live generation and outcome judging use Anthropic models. Cross-family self-enhancement validation remains unmeasured; reference labels are not blind human labels.
- Six original supported answers and eight adversarial pairs are diagnostic evidence. Zero observed failures does not establish universal robustness or justify autonomous release gating.
- The balanced diagnostic changes candidate slots, so its A/B slot win counts are not a stable generator-config comparison. The real legacy/hardened generation comparison tied on this diagnostic suite.
- The balanced live run predates the final stop-reason audit-field addition; that code is covered by regression tests. Measured files retain their original execution provenance.
- The submission document includes all six reflection answers and AI disclosures. The three Problem 1 responses were supplied by the candidate and inserted into the matching sections.
