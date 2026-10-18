"""How does sqlite-vec behave as the index grows?

Synthetic, clustered unit vectors (384-d, same as bge-small) at a few sizes.
For each size we measure insert throughput, file size and k=5 query latency for:

  float32  exact brute-force KNN (what the service uses today)
  float32 + metadata filter (collection = one of 10 values, ~10% of rows)
  int8     scalar-quantised vectors (scaled so the 99.9th pct component maps to 1), brute force
  bit      binary-quantised vectors, hamming KNN for 100 candidates, then exact
           cosine re-scoring of those 100 against float32 blobs in a plain table

Recall@5 for the quantised variants is measured against exact numpy search.
Brute-force latency doesn't depend on the data, but quantisation recall does:
real embeddings are usually friendlier than this synthetic set.

    python scripts/bench_store.py --sizes 100000 1000000
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import sqlite_vec

K = 5
RESCORE_CANDIDATES = 100


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = OFF")
    return db


def make_data(n: int, dim: int, n_queries: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((max(10, n // 100), dim)).astype(np.float32)
    assign = rng.integers(0, len(centroids), n)
    data = centroids[assign] + 0.6 * rng.standard_normal((n, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    picks = rng.choice(n, n_queries, replace=False)
    # perturb real points so every query has genuine near neighbours (noise norm ~0.5)
    queries = data[picks] + (0.5 / np.sqrt(dim)) * rng.standard_normal((n_queries, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return data, queries.astype(np.float32)


def exact_topk(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    out = []
    for start in range(0, len(queries), 25):
        scores = queries[start:start + 25] @ data.T
        idx = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, idx, axis=1).argsort(axis=1)[:, ::-1]
        out.append(np.take_along_axis(idx, order, axis=1))
    return np.vstack(out) + 1  # rowids start at 1


def timed_queries(db, sql: str, queries: np.ndarray, extra=()):
    latencies, results = [], []
    db.execute(sql, (queries[0].tobytes(), *extra)).fetchall()  # warm-up
    for q in queries:
        t0 = time.perf_counter()
        rows = db.execute(sql, (q.tobytes(), *extra)).fetchall()
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append([r[0] for r in rows])
    return latencies, results


def stats(latencies):
    return {"p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95)),
            "mean_ms": float(np.mean(latencies))}


def recall(found, truth):
    return float(np.mean([len(set(f[:K]) & set(t[:K])) / K for f, t in zip(found, truth)]))


def bench_size(n: int, dim: int, n_queries: int, workdir: Path) -> dict:
    data, queries = make_data(n, dim, n_queries)
    truth = exact_topk(data, queries, K)
    # int8 'unit' quantisation assumes components in [-1, 1]; unit vectors in 384-d have
    # components around +-0.05, so stretch them first or almost all precision is lost
    scale = float(1.0 / np.quantile(np.abs(data), 0.999))
    scaled = np.clip(data * scale, -1, 1).astype(np.float32)
    scaled_queries = np.clip(queries * scale, -1, 1).astype(np.float32)
    collections = [f"c{i % 10}" for i in range(n)]
    result = {"n": n, "dim": dim, "queries": n_queries}

    variants = {
        "float32": f"embedding float[{dim}] distance_metric=cosine, collection text",
        "int8": f"embedding int8[{dim}]",
        "bit": f"embedding bit[{dim}]",
    }
    inserts = {
        "float32": "INSERT INTO v(rowid, embedding, collection) VALUES (?, ?, ?)",
        "int8": "INSERT INTO v(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
        "bit": "INSERT INTO v(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
    }
    dbs = {}
    for name, columns in variants.items():
        path = workdir / f"bench_{name}_{n}.db"
        for p in workdir.glob(path.name + "*"):
            p.unlink()
        db = connect(path)
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0({columns})")
        if name == "bit":
            db.execute("CREATE TABLE raw (id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)")
        t0 = time.perf_counter()
        batch = 20_000
        for start in range(0, n, batch):
            rows = []
            for i in range(start, min(start + batch, n)):
                if name == "float32":
                    rows.append((i + 1, data[i].tobytes(), collections[i]))
                elif name == "int8":
                    rows.append((i + 1, scaled[i].tobytes()))
                else:
                    rows.append((i + 1, data[i].tobytes()))
            with db:
                db.executemany(inserts[name], rows)
                if name == "bit":
                    db.executemany("INSERT INTO raw(id, embedding) VALUES (?, ?)",
                                   [(i + 1, data[i].tobytes()) for i in range(start, min(start + batch, n))])
        elapsed = time.perf_counter() - t0
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = sum(p.stat().st_size for p in workdir.glob(path.name + "*"))
        result[name] = {"insert_seconds": elapsed, "inserts_per_second": n / elapsed, "file_mb": size / 1e6}
        dbs[name] = db

    f32 = dbs["float32"]
    lat, found = timed_queries(f32, "SELECT rowid FROM v WHERE embedding MATCH ? AND k = 5", queries)
    result["float32"].update(stats(lat), recall_at_5=recall(found, truth))

    lat, _ = timed_queries(f32, "SELECT rowid FROM v WHERE embedding MATCH ? AND k = 5 AND collection = ?",
                           queries, extra=("c3",))
    result["float32_filtered"] = stats(lat)

    lat, found = timed_queries(dbs["int8"],
                               "SELECT rowid FROM v WHERE embedding MATCH vec_quantize_int8(?, 'unit') AND k = 5",
                               scaled_queries)
    result["int8"].update(stats(lat), recall_at_5=recall(found, truth))

    # binary candidates, then exact re-scoring against float32 vectors
    bit = dbs["bit"]
    lat_total, lat_hamming, found = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        cands = [r[0] for r in bit.execute(
            f"SELECT rowid FROM v WHERE embedding MATCH vec_quantize_binary(?) AND k = {RESCORE_CANDIDATES}",
            (q.tobytes(),))]
        t1 = time.perf_counter()
        marks = ",".join("?" * len(cands))
        rows = bit.execute(
            f"SELECT id, vec_distance_cosine(embedding, ?) AS d FROM raw WHERE id IN ({marks}) "
            f"ORDER BY d LIMIT {K}", (q.tobytes(), *cands)).fetchall()
        lat_total.append((time.perf_counter() - t0) * 1000)
        lat_hamming.append((t1 - t0) * 1000)
        found.append([r[0] for r in rows])
    bit_index_mb = (n * dim / 8) / 1e6
    result["bit"].update(stats(lat_total), recall_at_5=recall(found, truth),
                         rescore_candidates=RESCORE_CANDIDATES,
                         hamming_p50_ms=float(np.percentile(lat_hamming, 50)),
                         bit_vectors_mb=bit_index_mb,
                         note="file_mb includes the float32 blobs used for re-scoring")

    for db in dbs.values():
        db.close()
    for p in workdir.glob(f"bench_*_{n}.db*"):
        p.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--out", default="results/bench_store.json")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="vecbench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass
    report = {"sqlite_vec": None, "cpu": cpu, "cpu_count": os.cpu_count(), "python": platform.python_version(),
              "k": K, "results": []}
    probe = connect(workdir / "probe.db")
    report["sqlite_vec"] = probe.execute("SELECT vec_version()").fetchone()[0]
    probe.close()

    for n in args.sizes:
        print(f"benchmarking {n:,} vectors ...", flush=True)
        row = bench_size(n, args.dim, args.queries, workdir)
        report["results"].append(row)
        print(json.dumps(row, indent=1), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
