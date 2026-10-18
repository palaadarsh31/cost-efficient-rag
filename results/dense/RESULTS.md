# Evaluation results

Generated 2026-10-18T02:32:30+00:00 at commit `53795e8` (docqa 0.1.0, Python 3.13.16).

- Embeddings: `BAAI/bge-small-en-v1.5` (384-d), stored as float32 in sqlite-vec
- Chunking: 350 tokens, 50 overlap; k = 5; default mode `dense`; similarity floor 0.5
- Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz (2 cores), 7.8 GB RAM, Linux-6.18.44-fc-v139-x86_64-with-glibc2.39

## Ingestion

| | value |
|---|---|
| documents (by type) | 33 (html: 10, md: 21, pdf: 2) |
| chunks / vectors / FTS rows | 274 / 274 / 274 |
| chunk tokens mean (min-max) | 251 (38-350) |
| index size on disk | 3.08 MB |
| first ingest | 33 added, 274 chunks, 44.0 s |
| re-ingest (same corpus) | 33 unchanged, 0 chunks written, vectors 274 -> 274, 0.01 s |
| idempotent | yes |

## Retrieval

Answerable questions only. A chunk is relevant if it contains at least half of a gold
evidence quote; recall is over evidence facts. 95% bootstrap CIs in brackets.

**dense** (25 questions)

| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |
|---|---|---|---|---|---|---|
| 1 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 |
| 3 | 0.920 | 0.920 | 0.740 | 0.774 | 0.320 | 0.733 |
| 5 | 0.960 [0.880, 1.000] | 0.960 [0.880, 1.000] | 0.750 [0.620, 0.870] | 0.799 [0.687, 0.896] | 0.208 [0.176, 0.248] | 0.740 [0.611, 0.860] |
| 10 | 0.960 | 0.960 | 0.750 | 0.799 | 0.104 | 0.740 |

**hybrid** (25 questions)

| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |
|---|---|---|---|---|---|---|
| 1 | 0.760 | 0.760 | 0.760 | 0.760 | 0.760 | 0.760 |
| 3 | 0.960 | 0.960 | 0.860 | 0.874 | 0.333 | 0.853 |
| 5 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.868 [0.764, 0.948] | 0.898 [0.823, 0.959] | 0.216 [0.200, 0.248] | 0.860 [0.760, 0.940] |
| 10 | 1.000 | 1.000 | 0.868 | 0.898 | 0.108 | 0.860 |

## Retrieval latency

Query embedding + vector/keyword search + chunk fetch, measured in-process, warm, each question repeated.

| mode | k | samples | embed p50 | search p50 | total p50 | total p95 | total max |
|---|---|---|---|---|---|---|---|
| dense | 5 | 150 | 9.95 | 0.60 | 10.82 | 14.65 | 23.50 |
| hybrid | 5 | 150 | 9.36 | 1.30 | 10.82 | 13.46 | 16.28 |

## Similarity floor (no-context gate)

Lowest top-1 similarity among answerable questions: 0.682; highest among unanswerable: 0.722. Highest floor that keeps every answerable question: 0.65.

| threshold | answerable kept | unanswerable blocked |
|---|---|---|
| 0.10 | 1.00 | 0.00 |
| 0.15 | 1.00 | 0.00 |
| 0.20 | 1.00 | 0.00 |
| 0.25 | 1.00 | 0.00 |
| 0.30 | 1.00 | 0.00 |
| 0.35 | 1.00 | 0.00 |
| 0.40 | 1.00 | 0.00 |
| 0.45 | 1.00 | 0.20 |
| 0.50 | 1.00 | 0.20 |
| 0.55 | 1.00 | 0.20 |
| 0.60 | 1.00 | 0.20 |
| 0.65 | 1.00 | 0.60 |
| 0.70 | 0.88 | 0.80 |
| 0.75 | 0.60 | 1.00 |
| 0.80 | 0.28 | 1.00 |
| 0.85 | 0.08 | 1.00 |
| 0.90 | 0.08 | 1.00 |

## Answer quality

Generator `claude-haiku-5-5`, judge `claude-sonnet-5-5`. Judge JSON parse failures: 0.

| metric | value [95% CI] | n | notes |
|---|---|---|---|
| faithfulness | 1.000 [1.000, 1.000] | 25 | supported claims / claims, over all answered questions |
| answer relevance | 0.970 [0.940, 1.000] | 25 | judge 1-5 rescaled to 0-1 |
| correctness vs gold | 0.980 [0.940, 1.000] | 25 | judge: correct 1 / partial 0.5 / incorrect 0; {'correct': 24, 'partial': 1} |
| exact match | 0.000 [0.000, 0.000] | 25 | SQuAD normalisation, citations stripped |
| token F1 | 0.275 [0.211, 0.338] | 25 | |
| contains gold | 0.760 [0.600, 0.920] | 25 | normalised gold string appears in answer |

| abstention / citations | value |
|---|---|
| unanswerable correctly declined | 5 / 5 |
| answerable wrongly declined | 0 / 25 |
| declined by similarity gate / by model | 1 / 4 |
| answers with >=1 valid citation | 25 / 25 |
| invalid citation markers | 0 |
| cited-chunk precision | 0.793 [0.673, 0.900] |

| latency (ms) | p50 | p95 | n |
|---|---|---|---|
| end-to-end, all questions | 895.3 | 1874.3 | 30 |
| end-to-end, LLM called | 928.5 | 1882.2 | 29 |
| LLM call only | 912.5 | 1866.6 | 29 |

Tokens per generation call: 2542 prompt + 101 completion (totals 73712 + 2931 over 29 calls). Judge used 81531 tokens.

_Total harness time: 181.9 s._
