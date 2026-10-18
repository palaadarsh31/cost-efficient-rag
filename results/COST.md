# Monthly cost: sqlite-vec (self-hosted) vs Pinecone

Assumptions: 384-d vectors, chunk text kept with the vector, 100,000 queries/month, 10% of chunks re-upserted per month, AWS us-east-1 on-demand. Full list in `scripts/cost_model.py`.

| vectors | sqlite-vec self-hosted | Pinecone serverless (Standard) | Pinecone pods (legacy) | notes |
|---|---|---|---|---|
| 100,000 | $14 (t4g.small, 20 GB gp3, db 0.6 GB) | $50 (usage $1: storage $0.10, reads $0.50, writes $0.13; 0.31 RU/query) | $81 (1x s1) / $81 (1x p1) | float32 brute force; measured: float32 p50 87 ms, binary+rescore p50 14 ms (recall@5 1.00) |
| 1,000,000 | $26 (t4g.medium, 20 GB gp3, db 6.1 GB) | $50 (usage $7: storage $1.04, reads $5.02, writes $1.26; 3.14 RU/query) | $81 (1x s1) / $162 (2x p1) | binary + float32 re-score; measured: float32 p50 917 ms, binary+rescore p50 147 ms (recall@5 1.00) |
| 10,000,000 | $221 (c7g.2xlarge, 106 GB gp3, db 61.0 GB) | $73 (usage $73: storage $10.36, reads $50.22, writes $12.56; 31.39 RU/query) | $243 (3x s1) / $891 (11x p1) | binary + re-score, 8 shards in parallel; extrapolated (linear from 1,000,000): binary scan ~1420 ms single-threaded, ~178 ms per query across 8 shards |

Sensitivity at 10M vectors (Pinecone serverless reads scale with namespace size x queries):

| queries / month | Pinecone serverless | sqlite-vec self-hosted |
|---|---|---|
| 10,000 | $50 | $221 (fixed) |
| 100,000 | $73 | $221 (fixed) |
| 1,000,000 | $525 | $221 (fixed) |
| 5,000,000 | $2534 | $221 (fixed) |
