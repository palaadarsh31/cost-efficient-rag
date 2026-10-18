# Evaluation results

> **Offline smoke run** (hash embedder + stub LLM). These numbers only show the
> harness works end to end; they are not quality measurements.

Generated 2026-10-18T06:08:24+00:00 at commit `None` (docqa 0.1.0, Python 3.12.10).

- Embeddings: `hash-bow-384` (384-d), stored as float32 in sqlite-vec
- Chunking: 350 tokens, 50 overlap; k = 5; default mode `hybrid`; similarity floor 0.5
- Hardware: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel (22 cores), RAM n/a, Windows-11-10.0.26200-SP0

## Ingestion

| | value |
|---|---|
| documents (by type) | 33 (html: 10, md: 21, pdf: 2) |
| chunks / vectors / FTS rows | 274 / 274 / 274 |
| chunk tokens mean (min-max) | 251 (38-350) |
| index size on disk | 3.08 MB |
| first ingest | 33 added, 274 chunks, 4.1 s |
| re-ingest (same corpus) | 33 unchanged, 0 chunks written, vectors 274 -> 274, 0.06 s |
| idempotent | yes |

## Retrieval

Answerable questions only. A chunk is relevant if it contains at least half of a gold
evidence quote; recall is over evidence facts. 95% bootstrap CIs in brackets.

**dense** (25 questions)

| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |
|---|---|---|---|---|---|---|
| 1 | 0.440 | 0.440 | 0.440 | 0.440 | 0.440 | 0.440 |
| 3 | 0.640 | 0.640 | 0.533 | 0.540 | 0.213 | 0.533 |
| 5 | 0.720 [0.520, 0.880] | 0.720 [0.520, 0.880] | 0.553 [0.380, 0.720] | 0.574 [0.406, 0.733] | 0.144 [0.104, 0.176] | 0.553 [0.380, 0.720] |
| 10 | 0.840 | 0.840 | 0.572 | 0.615 | 0.084 | 0.572 |

**hybrid** (25 questions)

| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |
|---|---|---|---|---|---|---|
| 1 | 0.640 | 0.640 | 0.640 | 0.640 | 0.640 | 0.640 |
| 3 | 0.840 | 0.840 | 0.733 | 0.740 | 0.280 | 0.733 |
| 5 | 0.880 [0.760, 1.000] | 0.880 [0.760, 1.000] | 0.743 [0.600, 0.880] | 0.764 [0.626, 0.888] | 0.184 [0.152, 0.216] | 0.731 [0.588, 0.868] |
| 10 | 0.920 | 0.920 | 0.748 | 0.777 | 0.096 | 0.736 |

## Retrieval latency

Query embedding + vector/keyword search + chunk fetch, measured in-process, warm, each question repeated.

| mode | k | samples | embed p50 | search p50 | total p50 | total p95 | total max |
|---|---|---|---|---|---|---|---|
| dense | 5 | 30 | 0.11 | 1.98 | 2.23 | 2.52 | 2.67 |
| hybrid | 5 | 30 | 0.12 | 3.16 | 3.43 | 5.17 | 5.56 |

## Similarity floor (no-context gate)

Lowest top-1 similarity among answerable questions: 0.146; highest among unanswerable: 0.226. Highest floor that keeps every answerable question: 0.10.

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.10 | 1.00 | 0.00 |
| 0.15 | 0.96 | 0.20 |
| 0.20 | 0.88 | 0.80 |
| 0.25 | 0.68 | 1.00 |
| 0.30 | 0.56 | 1.00 |
| 0.35 | 0.40 | 1.00 |
| 0.40 | 0.24 | 1.00 |
| 0.45 | 0.20 | 1.00 |
| 0.50 | 0.04 | 1.00 |
| 0.55 | 0.04 | 1.00 |
| 0.60 | 0.04 | 1.00 |
| 0.65 | 0.00 | 1.00 |
| 0.70 | 0.00 | 1.00 |
| 0.75 | 0.00 | 1.00 |
| 0.80 | 0.00 | 1.00 |
| 0.85 | 0.00 | 1.00 |
| 0.90 | 0.00 | 1.00 |

## Answer quality

Generator `offline-stub`, judge `offline-stub`. Judge JSON parse failures: 0.

Generation failures: 0. Failures are not counted as correct abstentions.

| metric | value [95% CI] | n | notes |
|---|---|---|---|
| faithfulness | 1.000 | 1 | supported claims / claims, over all answered questions |
| answer relevance | 0.750 | 1 | judge 1-5 rescaled to 0-1 |
| correctness vs gold | 0.020 [0.000, 0.060] | 25 | judge: correct 1 / partial 0.5 / incorrect 0; {'abstained': 24, 'partial': 1} |
| exact match | 0.000 [0.000, 0.000] | 25 | SQuAD-style; punctuation replaced with spaces, citations stripped |
| token F1 | 0.035 [0.000, 0.104] | 25 | |
| contains gold | 0.040 [0.000, 0.120] | 25 | normalised gold string appears in answer |

| abstention / citations | value |
|---|---|
| unanswerable correctly declined | 5 / 5 |
| answerable wrongly declined | 24 / 25 |
| declined by similarity gate / by model | 29 / 0 |
| answers with >=1 valid citation | 1 / 1 |
| invalid citation markers | 0 |
| gold-evidence citation precision | 1.000 |

Gold-evidence citation precision tests overlap with labeled answer evidence, not claim-to-citation entailment; extra supported claims can cite chunks outside that label set.

| latency (ms) | p50 | p95 | n |
|---|---|---|---|
| end-to-end, all questions | 3.6 | 5.3 | 30 |
| end-to-end, LLM called | 4.8 | 4.8 | 1 |
| LLM call only | 0.0 | 0.0 | 1 |

Tokens per generation call: 1517 prompt + 18 completion (totals 1517 + 18 over 1 calls, estimated). Judge used 2054 tokens.

_Total harness time: 8.1 s._
