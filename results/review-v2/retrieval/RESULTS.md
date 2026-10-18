# Evaluation results

Generated 2026-10-18T05:53:47+00:00 at commit `None` (docqa 0.1.0, Python 3.12.10).

- Embeddings: `BAAI/bge-small-en-v1.5` (384-d), stored as float32 in sqlite-vec
- Chunking: 350 tokens, 50 overlap; k = 5; default mode `hybrid`; similarity floor 0.5
- Hardware: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel (22 cores), RAM n/a, Windows-11-10.0.26200-SP0

## Ingestion

| | value |
|---|---|
| documents (by type) | 33 (html: 10, md: 21, pdf: 2) |
| chunks / vectors / FTS rows | 274 / 274 / 274 |
| chunk tokens mean (min-max) | 251 (38-350) |
| index size on disk | 3.08 MB |
| first ingest | 33 added, 274 chunks, 29.3 s |
| re-ingest (same corpus) | 33 unchanged, 0 chunks written, vectors 274 -> 274, 0.08 s |
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

## Embedding tokenizer audit

Actual model-token limit: 512; truncated inputs: 0/274; discarded tokens: 0. Title/section headers and special tokens are included.

| Chunk size/overlap | Inputs truncated by embedding tokenizer |
|---|---:|
| 128/16 | 0/703 |
| 256/32 | 0/362 |
| 350/0 | 0/270 |
| 350/50 | 0/274 |
| 512/64 | 33/195 |
| 800/100 | 83/135 |

## Retrieval latency

Query embedding + vector/keyword search + chunk fetch, measured in-process, warm, each question repeated.

| mode | k | samples | embed p50 | search p50 | total p50 | total p95 | total max |
|---|---|---|---|---|---|---|---|
| dense | 5 | 90 | 13.50 | 3.52 | 17.40 | 22.94 | 25.18 |
| hybrid | 5 | 90 | 12.60 | 4.78 | 17.88 | 24.36 | 37.72 |

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

## Chunking ablation

| size / overlap | chunks | mean tokens | dense Recall@5 | dense MRR@5 | dense nDCG@5 | dense P@5 | hybrid Recall@5 | hybrid MRR@5 | index MB |
|---|---|---|---|---|---|---|---|---|---|
| 128 / 16 | 703 | 97 | 0.840 | 0.701 | 0.726 | 0.184 | 0.960 | 0.748 | 3.18 |
| 256 / 32 | 362 | 189 | 0.960 | 0.723 | 0.783 | 0.208 | 0.960 | 0.753 | 3.06 |
| 350 / 0 | 270 | 250 | 0.960 | 0.737 | 0.790 | 0.208 | 1.000 | 0.868 | 3.08 |
| 350 / 50 | 274 | 251 | 0.960 | 0.750 | 0.799 | 0.208 | 1.000 | 0.868 | 3.08 |
| 512 / 64 | 195 | 350 | 0.920 | 0.780 | 0.817 | 0.200 | 0.960 | 0.848 | 3.20 |
| 800 / 100 | 135 | 503 | 0.960 | 0.775 | 0.818 | 0.200 | 0.960 | 0.880 | 3.08 |

## Answer quality

Not run (`--skip-answers`).

_Total harness time: 230.8 s._
