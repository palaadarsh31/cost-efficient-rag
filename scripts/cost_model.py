"""Monthly cost: self-hosted sqlite-vec vs a managed vector DB, at 100K / 1M / 10M chunks.

Every input is listed in ASSUMPTIONS with its source; change them and re-run.
Latency notes come from results/bench_store.json (scripts/bench_store.py).

    python scripts/cost_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOURS_PER_MONTH = 730

ASSUMPTIONS = {
    "dim": 384,  # BAAI/bge-small-en-v1.5
    "chunk_text_bytes": 1383,  # measured: mean chunk text in our corpus (350-token chunks)
    # measured with SQLite's dbstat on the full corpus index (scaled per chunk):
    # vector 1536 B + chunks row ~1840 B + FTS5 content ~1800 B + FTS5 index ~690 B + vec metadata ~200 B
    "sqlite_bytes_per_chunk": 6100,
    "pinecone_id_bytes": 20,
    "pinecone_metadata_bytes": 1383 + 200,  # chunk text stored as metadata + doc fields
    "queries_per_month": 100_000,  # "lightly queried": ~2.3 queries/minute on average
    "monthly_churn": 0.10,  # fraction of chunks re-upserted each month
    "upsert_batch": 100,
    "aws": {  # on-demand, us-east-1, Linux (aws.amazon.com/ec2/pricing/on-demand, /ebs/pricing, /s3/pricing)
        "t4g.small": {"usd_hour": 0.0168, "vcpu": 2, "ram_gb": 2},
        "t4g.medium": {"usd_hour": 0.0336, "vcpu": 2, "ram_gb": 4},
        "c7g.2xlarge": {"usd_hour": 0.289, "vcpu": 8, "ram_gb": 16},
        "r7g.xlarge": {"usd_hour": 0.2142, "vcpu": 4, "ram_gb": 32},
        "gp3_usd_gb_month": 0.08,
        "s3_usd_gb_month": 0.023,
    },
    "pinecone": {  # pinecone.io/pricing, docs.pinecone.io/guides/organizations/manage-cost/understanding-cost
        "standard_minimum_usd": 50.0,  # official pricing page, checked 2026-10-17
        "storage_usd_gb_month": 0.33,  # Standard, as reported in third-party 2026 summaries (calculator not machine-readable)
        "read_unit_usd_per_m": 16.0,  # idem
        "write_unit_usd_per_m": 4.0,  # idem
        "ru_per_gb_namespace": 1.0,  # official: 1 RU per GB of namespace, min 0.25 per query
        "ru_min_per_query": 0.25,
        "wu_per_kb": 1.0,  # official: 1 WU per KB of request, min 5 per request
        "pod_usd_hour": {"s1.x1": 0.1110, "p1.x1": 0.1110},  # pinecone.io/pricing/pods (legacy pod indexes)
        "pod_capacity_768d": {"s1.x1": 5_000_000, "p1.x1": 1_000_000},  # vectors, 768-d, no metadata
    },
}

# Instance chosen per scale and why (latency from scripts/bench_store.py on a 2-vCPU box)
SELF_HOSTED_PLAN = {
    100_000: {"instance": "t4g.small", "shards": 1, "search": "float32 brute force"},
    1_000_000: {"instance": "t4g.medium", "shards": 1, "search": "binary + float32 re-score"},
    10_000_000: {"instance": "c7g.2xlarge", "shards": 8, "search": "binary + re-score, 8 shards in parallel"},
}


def self_hosted(n: int, a=ASSUMPTIONS) -> dict:
    plan = SELF_HOSTED_PLAN[n]
    aws = a["aws"]
    inst = aws[plan["instance"]]
    db_gb = n * a["sqlite_bytes_per_chunk"] / 1e9
    disk_gb = max(20, round(db_gb * 1.6 + 8))  # db + WAL/rebuild headroom + OS
    compute = inst["usd_hour"] * HOURS_PER_MONTH
    disk = disk_gb * aws["gp3_usd_gb_month"]
    backup = db_gb * aws["s3_usd_gb_month"]
    return {"plan": plan, "db_gb": db_gb, "disk_gb": disk_gb, "compute": compute, "disk": disk,
            "backup": backup, "total": compute + disk + backup}


def pinecone_serverless(n: int, a=ASSUMPTIONS) -> dict:
    p = a["pinecone"]
    record_bytes = a["pinecone_id_bytes"] + a["pinecone_metadata_bytes"] + a["dim"] * 4
    gb = n * record_bytes / 1e9
    storage = gb * p["storage_usd_gb_month"]
    ru_per_query = max(p["ru_min_per_query"], gb * p["ru_per_gb_namespace"])
    reads = a["queries_per_month"] * ru_per_query * p["read_unit_usd_per_m"] / 1e6
    upserted = n * a["monthly_churn"]
    wu = upserted * record_bytes / 1000 * p["wu_per_kb"]
    wu = max(wu, 5 * upserted / a["upsert_batch"])
    writes = wu * p["write_unit_usd_per_m"] / 1e6
    usage = storage + reads + writes
    return {"index_gb": gb, "ru_per_query": ru_per_query, "storage": storage, "reads": reads,
            "writes": writes, "usage": usage, "total": max(usage, p["standard_minimum_usd"])}


def pinecone_pods(n: int, pod: str, a=ASSUMPTIONS) -> dict:
    p = a["pinecone"]
    # capacity is quoted for 768-d vectors without metadata (~3 KB each); scale by bytes per record
    quoted_bytes = 768 * 4
    record_bytes = a["pinecone_id_bytes"] + a["pinecone_metadata_bytes"] + a["dim"] * 4
    capacity = int(p["pod_capacity_768d"][pod] * quoted_bytes / record_bytes)
    pods = max(1, -(-n // capacity))
    return {"pod": pod, "pods": pods, "capacity_per_pod": capacity,
            "total": pods * p["pod_usd_hour"][pod] * HOURS_PER_MONTH}


def main() -> None:
    bench_path = ROOT / "results" / "bench_store.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
    latency = {}
    if bench:
        for row in bench["results"]:
            latency[row["n"]] = row

    rows = []
    for n in (100_000, 1_000_000, 10_000_000):
        ours, serverless = self_hosted(n), pinecone_serverless(n)
        s1, p1 = pinecone_pods(n, "s1.x1"), pinecone_pods(n, "p1.x1")
        rows.append({"vectors": n, "self_hosted": ours, "pinecone_serverless": serverless,
                     "pinecone_s1": s1, "pinecone_p1": p1})

    out = {"assumptions": ASSUMPTIONS, "self_hosted_plan": SELF_HOSTED_PLAN, "rows": rows,
           "bench_latency": latency}
    (ROOT / "results" / "cost_model.json").write_text(json.dumps(out, indent=2))

    def lat_note(n):
        if n in latency:
            r = latency[n]
            return (f"measured: float32 p50 {r['float32']['p50_ms']:.0f} ms, "
                    f"binary+rescore p50 {r['bit']['p50_ms']:.0f} ms (recall@5 {r['bit']['recall_at_5']:.2f})")
        if latency:
            biggest = max(latency)
            r = latency[biggest]
            factor = n / biggest
            per_shard = r["bit"]["hamming_p50_ms"] * factor / SELF_HOSTED_PLAN[n]["shards"]
            return (f"extrapolated (linear from {biggest:,}): binary scan ~{r['bit']['hamming_p50_ms'] * factor:.0f} ms "
                    f"single-threaded, ~{per_shard:.0f} ms per query across {SELF_HOSTED_PLAN[n]['shards']} shards")
        return "run scripts/bench_store.py for latency"

    lines = [
        "# Monthly cost: sqlite-vec (self-hosted) vs Pinecone", "",
        f"Assumptions: {ASSUMPTIONS['dim']}-d vectors, chunk text kept with the vector, "
        f"{ASSUMPTIONS['queries_per_month']:,} queries/month, {int(ASSUMPTIONS['monthly_churn'] * 100)}% of chunks "
        "re-upserted per month, AWS us-east-1 on-demand. Full list in `scripts/cost_model.py`.", "",
        "| vectors | sqlite-vec self-hosted | Pinecone serverless (Standard) | Pinecone pods (legacy) | notes |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        n, o, s = r["vectors"], r["self_hosted"], r["pinecone_serverless"]
        lines.append(
            f"| {n:,} | ${o['total']:.0f} ({o['plan']['instance']}, {o['disk_gb']} GB gp3, db {o['db_gb']:.1f} GB) | "
            f"${s['total']:.0f} (usage ${s['usage']:.0f}: storage ${s['storage']:.2f}, reads ${s['reads']:.2f}, "
            f"writes ${s['writes']:.2f}; {s['ru_per_query']:.2f} RU/query) | "
            f"${r['pinecone_s1']['total']:.0f} ({r['pinecone_s1']['pods']}x s1) / ${r['pinecone_p1']['total']:.0f} "
            f"({r['pinecone_p1']['pods']}x p1) | {o['plan']['search']}; {lat_note(n)} |")

    # how query volume moves the 10M comparison
    lines += ["", "Sensitivity at 10M vectors (Pinecone serverless reads scale with namespace size x queries):", "",
              "| queries / month | Pinecone serverless | sqlite-vec self-hosted |", "|---|---|---|"]
    for q in (10_000, 100_000, 1_000_000, 5_000_000):
        a = json.loads(json.dumps(ASSUMPTIONS))
        a["queries_per_month"] = q
        lines.append(f"| {q:,} | ${pinecone_serverless(10_000_000, a)['total']:.0f} | "
                     f"${self_hosted(10_000_000, a)['total']:.0f} (fixed) |")
    (ROOT / "results" / "COST.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
