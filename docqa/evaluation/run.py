"""Evaluation harness: retrieval, answer quality, latency, chunking ablation.

    python -m docqa.evaluation.run                     # everything (needs an LLM)
    python -m docqa.evaluation.run --skip-answers      # retrieval + latency only, no LLM calls
    python -m docqa.evaluation.run --offline           # plumbing check: hash embedder + stub LLM

Writes results/summary.json, per-question JSONL files and results/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import Settings
from ..embeddings import Embedder, HashEmbedder, build_embedder
from ..ingest import ingest_path
from ..llm import ChatClient, LLMError
from ..loaders import load_document
from ..qa import QAPipeline
from ..passages import FORMAT_VERSION, coerce_passages, format_passages
from ..retrieval import Retriever
from ..store import VectorStore
from . import metrics as M
from .judge import Judge
from .labels import Labels, Question, label_question, load_questions
from .offline import StubLLM
from .report import render_markdown

log = logging.getLogger("docqa.eval")

KS = (1, 3, 5, 10)
ABLATION_CONFIGS = [(128, 16), (256, 32), (350, 0), (350, 50), (512, 64), (800, 100)]


# --------------------------------------------------------------- helpers

def _round(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


def _summary_stats(values: list[float]) -> dict:
    return {
        "p50": M.percentile(values, 50),
        "p95": M.percentile(values, 95),
        "mean": M.mean(values),
        "max": max(values) if values else None,
        "n": len(values),
    }


def run_info(settings: Settings, embedder: Embedder, args) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            names = [ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")]
            cpu = names[0] if names else cpu
    except OSError:
        pass
    ram_gb = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                ram_gb = round(int(fh.readline().split()[1]) / 2**20, 1)
        except (OSError, ValueError, IndexError):
            pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docqa_version": __version__,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "embed_model": embedder.model_name,
        "embed_dim": embedder.dim,
        "offline": args.offline,
        "settings": settings.redacted(),
    }


def build_index(corpus: Path, db_path: Path, embedder: Embedder, settings: Settings):
    for p in db_path.parent.glob(db_path.name + "*"):
        p.unlink()
    store = VectorStore(db_path)
    first = ingest_path(corpus, store, embedder, settings)
    return store, first


def make_labels(store: VectorStore, corpus: Path, questions: list[Question]) -> dict[str, Labels]:
    root = Path(store.meta().get("corpus_root", corpus))
    texts = {doc_id: load_document(root / doc_id, root).text for doc_id in store.doc_ids()}
    return {q.id: label_question(q, store, texts) for q in questions if q.answerable}


def embedding_input_audit(store: VectorStore, embedder: Embedder) -> dict:
    audit = getattr(embedder, "audit_documents", None)
    if not callable(audit):
        return {"available": False, "reason": "Embedding backend has no tokenizer audit"}
    texts, identifiers = [], []
    for doc_id in store.doc_ids():
        title = store.get_document(doc_id)["title"]
        for chunk in store.chunks_for(doc_id):
            header = [title]
            if chunk.section and chunk.section not in title:
                header.append(chunk.section)
            texts.append(" | ".join(header) + "\n\n" + chunk.text)
            identifiers.append({"doc_id": doc_id, "chunk_id": chunk.chunk_id})
    result = audit(texts)
    for row, identity in zip(result.get("per_input", []), identifiers):
        row.update(identity)
    return result


# ------------------------------------------------------------- retrieval

def evaluate_retrieval(retriever: Retriever, questions: list[Question], labels: dict[str, Labels],
                       mode: str, ks=KS) -> dict:
    max_k = max(ks)
    per_question = []
    for q in questions:
        if not q.answerable:
            continue
        lab = labels[q.id]
        r = retriever.retrieve(q.question, k=max_k, mode=mode)
        ranked = [h.chunk.chunk_id for h in r.hits]
        first_rel = next((i for i, cid in enumerate(ranked, start=1) if cid in lab.relevant), None)
        row = {
            "id": q.id,
            "n_relevant": len(lab.relevant),
            "n_facts": len(lab.facts),
            "first_relevant_rank": first_rel,
            "top_docs": [h.chunk.doc_id for h in r.hits[:5]],
            "top_similarity": r.top_similarity,
        }
        for k in ks:
            row[f"hit@{k}"] = M.hit_at_k(ranked, lab.relevant, k)
            row[f"recall@{k}"] = M.recall_at_k(ranked, lab.facts, k)
            row[f"mrr@{k}"] = M.reciprocal_rank(ranked, lab.relevant, k)
            row[f"ndcg@{k}"] = M.ndcg_at_k(ranked, lab.relevant, k)
            row[f"precision@{k}"] = M.precision_at_k(ranked, lab.relevant, k)
            row[f"context_precision@{k}"] = M.context_precision_at_k(ranked, lab.relevant, k)
        per_question.append(row)

    names = ("hit", "recall", "mrr", "ndcg", "precision", "context_precision")
    by_k = {}
    for k in ks:
        by_k[k] = {}
        for name in names:
            vals = [row[f"{name}@{k}"] for row in per_question]
            ci = M.bootstrap_ci(vals)
            by_k[k][name] = {"mean": M.mean(vals), "ci95": list(ci) if ci else None}
    return {"mode": mode, "n_questions": len(per_question), "by_k": by_k, "per_question": per_question}


def measure_latency(retriever: Retriever, questions: list[Question], mode: str, k: int,
                    repeats: int) -> dict:
    retriever.retrieve("warm-up query", k=k, mode=mode)
    samples = {"embed": [], "search": [], "fetch": [], "total": []}
    for _ in range(repeats):
        for q in questions:
            t = retriever.retrieve(q.question, k=k, mode=mode).timings_ms
            for name in samples:
                samples[name].append(t[name])
    return {"mode": mode, "k": k, "repeats": repeats,
            **{name: _summary_stats(vals) for name, vals in samples.items()}}


def threshold_sweep(retriever: Retriever, questions: list[Question]) -> dict:
    rows = []
    for q in questions:
        r = retriever.retrieve(q.question, k=1, mode="dense")
        rows.append({"id": q.id, "answerable": q.answerable, "top_similarity": r.top_similarity})
    ans = [r["top_similarity"] for r in rows if r["answerable"]]
    unans = [r["top_similarity"] for r in rows if not r["answerable"]]
    sweep = []
    for t in np.round(np.arange(0.10, 0.91, 0.05), 2):
        sweep.append({
            "threshold": float(t),
            "answerable_kept": sum(s >= t for s in ans) / len(ans) if ans else None,
            "unanswerable_blocked": sum(s < t for s in unans) / len(unans) if unans else None,
        })
    keep_all = [row["threshold"] for row in sweep if row["answerable_kept"] == 1.0]
    return {"per_question": rows, "sweep": sweep,
            "answerable_min": min(ans) if ans else None, "unanswerable_max": max(unans) if unans else None,
            "highest_threshold_keeping_all_answerable": max(keep_all) if keep_all else None}


def chunking_ablation(corpus: Path, work: Path, embedder: Embedder, settings: Settings,
                      questions: list[Question], configs=ABLATION_CONFIGS) -> list[dict]:
    rows = []
    for size, overlap in configs:
        cfg = settings.copy(chunk_size=size, chunk_overlap=overlap)
        store, report = build_index(corpus, work / f"ablation_{size}_{overlap}.db", embedder, cfg)
        labels = make_labels(store, corpus, questions)
        retriever = Retriever(store, embedder, cfg)
        dense = evaluate_retrieval(retriever, questions, labels, "dense", ks=(5,))["by_k"][5]
        hybrid = evaluate_retrieval(retriever, questions, labels, "hybrid", ks=(5,))["by_k"][5]
        tokens = [c.n_tokens for d in store.doc_ids() for c in store.chunks_for(d)]
        rows.append({
            "chunk_size": size,
            "chunk_overlap": overlap,
            "n_chunks": len(tokens),
            "mean_chunk_tokens": float(np.mean(tokens)),
            "db_bytes": store.size_bytes(),
            "ingest_seconds": report.seconds,
            "ingestion_failures": report.failed,
            "embedding_input_audit": embedding_input_audit(store, embedder),
            "dense@5": {k: v["mean"] for k, v in dense.items()},
            "hybrid@5": {k: v["mean"] for k, v in hybrid.items()},
        })
        log.info("ablation %s/%s: %d chunks, dense recall@5 %.3f", size, overlap, len(tokens),
                 dense["recall"]["mean"])
        store.close()
    return rows


# --------------------------------------------------------------- answers

def evaluate_answers(pipeline: QAPipeline, judge: Judge, questions: list[Question],
                     labels: dict[str, Labels], out_rows: list[dict]) -> dict:
    for i, q in enumerate(questions, start=1):
        log.info("answering %s (%d/%d)", q.id, i, len(questions))
        generation_started = time.perf_counter()
        try:
            ans = pipeline.answer(q.question)
        except LLMError as exc:
            rejected = getattr(exc, "completion", None)
            elapsed_ms = (time.perf_counter() - generation_started) * 1000
            failure = {"id": q.id, "question": q.question, "answerable": q.answerable,
                             "gold": q.answer, "answer": None, "generation_failed": True,
                             "error": str(exc), "abstained": False, "abstain_reason": None,
                             "uncited": False, "cited": [], "invalid_citations": [], "retrieved": [],
                             "llm_called": True, "usage": {"prompt_tokens": getattr(rejected, "prompt_tokens", 0),
                             "completion_tokens": getattr(rejected, "completion_tokens", 0), "estimated": rejected is None},
                             "timings_ms": {"total": elapsed_ms, "llm": getattr(rejected, "latency_ms", elapsed_ms)},
                             "model": getattr(rejected, "model", None),
                             "generation_audit": getattr(exc, "generation_audit", {}),
                             "rejected_completion": rejected.__dict__ if rejected else None}
            if q.answerable:
                failure.update({"em": 0.0, "f1": 0.0, "contains_gold": 0.0, "correctness": 0.0})
            out_rows.append(failure)
            continue
        passages = coerce_passages(ans.retrieved)
        row = {
            "id": q.id,
            "question": q.question,
            "answerable": q.answerable,
            "gold": q.answer,
            "answer": ans.answer,
            "abstained": ans.abstained,
            "abstain_reason": ans.abstain_reason,
            "uncited": ans.uncited,
            "cited": [c.chunk_id for c in ans.citations],
            "invalid_citations": ans.invalid_citations,
            "retrieved": [{**h,
                           "relevant": (h["chunk_id"] in labels[q.id].relevant) if q.answerable else None}
                          for h in ans.retrieved],
            "passage_order": [p.citation_id for p in passages],
            "passage_formatter_version": FORMAT_VERSION,
            "generator_context": format_passages(passages),
            "generation_audit": ans.generation_audit,
            "generation_failed": False,
            "top_similarity": ans.top_similarity,
            "llm_called": ans.abstain_reason != "no_relevant_context",
            "usage": ans.usage,
            "timings_ms": ans.timings_ms,
            "model": ans.model,
        }
        if q.answerable:
            prediction = "" if ans.abstained else ans.answer
            row["em"] = M.exact_match(prediction, q.answer)
            row["f1"] = M.token_f1(prediction, q.answer)
            row["contains_gold"] = M.contains_gold(prediction, q.answer)
            cited = set(row["cited"])
            row["cited_precision"] = (len(cited & labels[q.id].relevant) / len(cited)) if cited else None
            row["gold_evidence_citation_precision"] = row["cited_precision"]
        if not ans.abstained:
            faith = judge.faithfulness(q.question, ans.answer, passages)
            row["faithfulness"] = faith.score
            row["judge_faithfulness"] = faith.__dict__
            if q.answerable:
                rel = judge.relevance(q.question, ans.answer)
                cor = judge.correctness(q.question, q.answer, ans.answer)
                row["relevance"] = rel.score
                row["correctness"] = cor.score
                row["judge_relevance"] = rel.__dict__
                row["judge_correctness"] = cor.__dict__
        elif q.answerable:
            row["correctness"] = 0.0  # declining an answerable question is a miss
        out_rows.append(row)

    answerable = [r for r in out_rows if r["answerable"]]
    unanswerable = [r for r in out_rows if not r["answerable"]]
    answered = [r for r in out_rows if not r["abstained"] and not r.get("generation_failed")]
    llm_called = [r for r in out_rows if r["llm_called"]]

    def stat(rows, key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        ci = M.bootstrap_ci(vals)
        return {"mean": M.mean(vals), "ci95": list(ci) if ci else None, "n": len(vals)}

    judge_rows = [r for r in out_rows if "judge_faithfulness" in r]
    judge_tokens = sum(r[f"judge_{n}"]["prompt_tokens"] + r[f"judge_{n}"]["completion_tokens"]
                       for r in judge_rows for n in ("faithfulness", "relevance", "correctness")
                       if f"judge_{n}" in r)
    judge_prompt_tokens = sum(r[f"judge_{n}"]["prompt_tokens"] for r in judge_rows
                             for n in ("faithfulness", "relevance", "correctness") if f"judge_{n}" in r)
    judge_completion_tokens = judge_tokens - judge_prompt_tokens
    parse_errors = sum(1 for r in judge_rows for n in ("faithfulness", "relevance", "correctness")
                       if f"judge_{n}" in r and r[f"judge_{n}"]["parse_error"])
    labels_count = {}
    for r in answerable:
        lab = (r.get("judge_correctness") or {}).get("detail", {}).get("label", "abstained" if r["abstained"] else "unparsed")
        labels_count[lab] = labels_count.get(lab, 0) + 1

    return {
        "n_questions": len(out_rows),
        "generation_failures": sum(r.get("generation_failed", False) for r in out_rows),
        "failures": [{"id": r["id"], "error": r.get("error")} for r in out_rows if r.get("generation_failed")],
        "faithfulness": stat(answered, "faithfulness"),
        "faithfulness_answerable_only": stat([r for r in answered if r["answerable"]], "faithfulness"),
        "relevance": stat(answerable, "relevance"),
        "correctness": stat(answerable, "correctness"),
        "correctness_labels": labels_count,
        "em": stat(answerable, "em"),
        "f1": stat(answerable, "f1"),
        "contains_gold": stat(answerable, "contains_gold"),
        "abstention": {
            "unanswerable_abstained": sum(r["abstained"] for r in unanswerable),
            "unanswerable_total": len(unanswerable),
            "answerable_abstained": sum(r["abstained"] for r in answerable),
            "answerable_total": len(answerable),
            "reasons": {reason: sum(1 for r in out_rows if r["abstain_reason"] == reason)
                        for reason in ("no_relevant_context", "model_abstained")},
        },
        "citations": {
            "answered": len(answered),
            "answered_with_citation": sum(1 for r in answered if r["cited"]),
            "uncited_answers": sum(1 for r in answered if r["uncited"]),
            "invalid_citation_markers": sum(len(r["invalid_citations"]) for r in out_rows),
            "cited_chunk_precision": stat([r for r in answerable if not r["abstained"]], "cited_precision"),
        },
        "latency_ms": {
            "end_to_end_all": _summary_stats([r["timings_ms"]["total"] for r in out_rows]),
            "end_to_end_llm_called": _summary_stats([r["timings_ms"]["total"] for r in llm_called]),
            "llm": _summary_stats([r["timings_ms"]["llm"] for r in llm_called]),
        },
        "tokens": {
            "llm_calls": len(llm_called),
            "prompt_mean": M.mean([r["usage"]["prompt_tokens"] for r in llm_called]),
            "completion_mean": M.mean([r["usage"]["completion_tokens"] for r in llm_called]),
            "prompt_total": sum(r["usage"]["prompt_tokens"] for r in llm_called),
            "completion_total": sum(r["usage"]["completion_tokens"] for r in llm_called),
            "estimated": any(r["usage"].get("estimated") for r in llm_called),
            "judge_total": judge_tokens,
            "judge_prompt_total": judge_prompt_tokens,
            "judge_completion_total": judge_completion_tokens,
            "n_user_queries": len(out_rows),
        },
        "judge_parse_errors": parse_errors,
        "generator_model": next((r["model"] for r in llm_called if r["model"]), None),
    }


def llm_cost(tokens: dict) -> dict | None:
    price_in = os.getenv("LLM_PRICE_INPUT_PER_M")
    price_out = os.getenv("LLM_PRICE_OUTPUT_PER_M")
    if not price_in or not price_out or not tokens.get("prompt_mean"):
        return None
    per_call = (tokens["prompt_mean"] * float(price_in) + tokens["completion_mean"] * float(price_out)) / 1e6
    n_queries = tokens.get("n_user_queries")
    total = (tokens["prompt_total"] * float(price_in) + tokens["completion_total"] * float(price_out)) / 1e6
    return {"usd_per_1k_queries": total * 1000 / n_queries if n_queries else None,
            "usd_per_1k_generation_calls": per_call * 1000, "total_generation_usd": total,
            "price_input_per_m": float(price_in),
            "price_output_per_m": float(price_out)}


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docqa.evaluation.run")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="results")
    parser.add_argument("--repeats", type=int, default=5, help="latency repetitions per question")
    parser.add_argument("--skip-answers", action="store_true", help="no LLM calls")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="hash embedder + stub LLM; checks the plumbing, numbers are meaningless")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    if args.offline:
        out = out / "offline-smoke"
        settings = settings.copy(embed_backend="hash")
    work = out / ".work"
    if (out / "summary.json").exists() or work.exists():
        parser.error("Evaluation output or working directory already exists; choose a fresh --out to preserve prior evidence")
    work.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)

    embedder = HashEmbedder() if args.offline else build_embedder(settings)
    questions = load_questions(args.questions)
    answerable = [q for q in questions if q.answerable]
    started = time.perf_counter()
    info = run_info(settings, embedder, args)
    info["evaluation_protocol"] = "shared-source-information-v2"
    log.info("evaluating %d questions (%d answerable) with %s", len(questions), len(answerable),
             embedder.model_name)

    # 1. index from scratch, then re-ingest to prove idempotency
    store, first = build_index(corpus, work / "eval.db", embedder, settings)
    second = ingest_path(corpus, store, embedder, settings)
    counts = store.counts()
    ingestion = {
        "first_run": first.as_dict(),
        "second_run": second.as_dict(),
        "idempotent": second.vectors_after == first.vectors_after and second.chunks_written == 0,
        "counts": counts,
        "db_bytes": store.size_bytes(),
    }
    tokens = [c.n_tokens for d in store.doc_ids() for c in store.chunks_for(d)]
    ingestion["chunk_tokens"] = {"mean": float(np.mean(tokens)), "min": int(min(tokens)), "max": int(max(tokens))}
    doc_types = {}
    for doc_id in store.doc_ids():
        dt = store.get_document(doc_id)["doc_type"]
        doc_types[dt] = doc_types.get(dt, 0) + 1
    ingestion["documents_by_type"] = doc_types
    ingestion["embedding_input_audit"] = embedding_input_audit(store, embedder)

    labels = make_labels(store, corpus, questions)
    retriever = Retriever(store, embedder, settings)

    # 2. retrieval quality, 3. latency, 4. abstention threshold
    retrieval = {mode: evaluate_retrieval(retriever, questions, labels, mode) for mode in ("dense", "hybrid")}
    latency = {mode: measure_latency(retriever, questions, mode, settings.top_k, args.repeats)
               for mode in ("dense", "hybrid")}
    sweep = threshold_sweep(retriever, questions)

    # 5. chunking ablation
    ablation = None if args.skip_ablation else chunking_ablation(corpus, work, embedder, settings, questions)

    # 6. answers
    answers, answer_rows = None, []
    if not args.skip_answers:
        if args.offline:
            llm = judge_client = StubLLM()
        else:
            llm, judge_client = ChatClient.for_generation(settings), ChatClient.for_judge(settings)
        pipeline = QAPipeline(retriever, llm, settings.copy(query_log_path=out / "query_log.jsonl"))
        answers = evaluate_answers(pipeline, Judge(judge_client), questions, labels, answer_rows)
        answers["judge_model"] = getattr(judge_client, "model", None)
        answers["llm_cost"] = llm_cost(answers["tokens"])

    summary = _round({
        "run": info,
        "ingestion": ingestion,
        "retrieval": {m: {"n_questions": r["n_questions"], "by_k": r["by_k"]} for m, r in retrieval.items()},
        "latency_ms": latency,
        "threshold": {k: v for k, v in sweep.items() if k != "per_question"},
        "ablation_chunking": ablation,
        "answers": answers,
        "run_status": "incomplete" if answers and (answers.get("generation_failures") or answers.get("judge_parse_errors"))
                      or first.failed or second.failed or not ingestion["idempotent"]
                      or any(row.get("ingestion_failures") for row in (ablation or [])) else "complete",
        "elapsed_seconds": time.perf_counter() - started,
    })

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "retrieval_per_question.jsonl", "w", encoding="utf-8") as fh:
        for mode, r in retrieval.items():
            for row in r["per_question"]:
                fh.write(json.dumps(_round({"mode": mode, **row})) + "\n")
    (out / "threshold_per_question.json").write_text(json.dumps(_round(sweep["per_question"]), indent=2))
    if answer_rows:
        with open(out / "answers.jsonl", "w", encoding="utf-8") as fh:
            for row in answer_rows:
                fh.write(json.dumps(_round(row), ensure_ascii=False) + "\n")
    (out / "RESULTS.md").write_text(render_markdown(summary), encoding="utf-8")
    store.close()
    shutil.rmtree(work, ignore_errors=True)

    print(f"wrote {out}/summary.json and {out}/RESULTS.md in {summary['elapsed_seconds']:.1f} s")
    if not ingestion["idempotent"]:
        print("WARNING: re-ingest changed the index")
        return 1
    return 0 if summary["run_status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
