"""Frozen-answer source-information and passage-order experiments.

This module makes no generation calls. Reconstruct saved chunks only after their
content IDs and legacy judge context match the immutable baseline.
"""
from __future__ import annotations

import hashlib
import math
import json
import random
from dataclasses import asdict
from pathlib import Path

from ..chunking import chunk_document
from ..loaders import load_document
from ..passages import FORMAT_VERSION, Passage, format_passages
from ..store import make_chunk_id


def fingerprint(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def restore_baseline_cases(answers_path: Path, corpus: Path, ids: list[str],
                           chunk_size: int = 350, overlap: int = 50) -> list[dict]:
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Choose nonempty unique baseline question IDs")
    rows = [json.loads(line) for line in answers_path.read_text(encoding="utf8").splitlines() if line.strip()]
    saved = {r["id"]: r for r in rows}
    if any(qid not in saved for qid in ids):
        raise ValueError("Unknown baseline question ID")
    cache = {}
    out = []
    for qid in ids:
        row = saved[qid]
        passages = []
        for n, hit in enumerate(row["retrieved"], 1):
            doc_id = hit["doc_id"]
            if doc_id not in cache:
                doc = load_document(corpus / doc_id, corpus)
                cache[doc_id] = (doc, {make_chunk_id(doc_id, c): c for c in chunk_document(doc, chunk_size, overlap)})
            doc, chunks = cache[doc_id]
            if hit["chunk_id"] not in chunks:
                raise ValueError(f"{qid}: baseline chunk ID cannot be recovered: {hit['chunk_id']}")
            c = chunks[hit["chunk_id"]]
            passages.append(Passage(n, hit["chunk_id"], doc_id, doc.title, c.text,
                                    c.section, c.page_start, c.page_end))
        legacy_context = "\n\n".join(f"[{p.citation_id}] {p.text}" for p in passages)
        old_prompt = row["judge_faithfulness"]["prompt"]
        original_context = old_prompt.split("Context passages:\n", 1)[1].rsplit("\n\nQuestion:", 1)[0]
        if legacy_context != original_context:
            raise ValueError(f"{qid}: recovered text differs from saved baseline context; refusing replay")
        out.append({
            "id": qid, "question": row["question"], "answer": row["answer"],
            "gold": row.get("gold"), "baseline_faithfulness": row.get("faithfulness"),
            "passages": [p.as_dict() for p in passages], "legacy_context": legacy_context,
            "source_formatter_version": FORMAT_VERSION,
            "baseline_row_sha256": fingerprint(row),
            "frozen_input_sha256": fingerprint([row["question"], row["answer"], [p.as_dict() for p in passages]]),
            "reconstruction": "Chunk IDs and exact text verified against saved judge prompt; metadata recovered from deterministic corpus loader.",
            "metadata_limitation": "Historical generator request headers were not archived; reconstructed metadata cannot be checked byte-for-byte against that request.",
        })
    return out


def permutations(passages: list[Passage], seed: int, shuffled: int = 4,
                 original_repeats: int = 4) -> list[dict]:
    if original_repeats < 2 or shuffled < 1:
        raise ValueError("Need repeated unchanged controls and at least one shuffle")
    original = tuple(p.citation_id for p in passages)
    if len(set(original)) != len(original):
        raise ValueError("Duplicate citations")
    if len(original) < 3:
        raise ValueError("At least three passages needed for distinct order controls")
    by_id = {p.citation_id: p for p in passages}
    jobs = [{"variant": f"original_{n}", "kind": "unchanged", "order": list(original)}
            for n in range(original_repeats)]
    reverse = tuple(reversed(original))
    jobs.append({"variant": "reversed", "kind": "reordered", "order": list(reverse)})
    if shuffled > math.factorial(len(original)) - 2:
        raise ValueError("Too many unique shuffled orders requested")
    rng = random.Random(seed)
    chosen = set((original, reverse))
    selected = []
    while len(selected) < shuffled:
        order = list(original)
        rng.shuffle(order)
        order = tuple(order)
        if order not in chosen:
            selected.append(order)
            chosen.add(order)
    for n, order in enumerate(selected):
        jobs.append({"variant": f"shuffled_{n}", "kind": "reordered", "order": list(order)})
    for job in jobs:
        job["passages"] = [by_id[citation_id] for citation_id in job["order"]]
    return jobs


def summarize_order(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    per_case = []
    all_control, all_changed = [], []
    failures = [r for r in rows if r.get("judgement", {}).get("score") is None]
    for case_id, trials in groups.items():
        reference = next(r for r in trials if r["variant"] == "original_0")
        reference_score = reference.get("judgement", {}).get("score")
        groups_delta = {"unchanged": [], "reordered": []}
        for row in trials:
            score = row.get("judgement", {}).get("score")
            if row is reference or score is None or reference_score is None:
                continue
            groups_delta[row["kind"]].append({
                "variant": row["variant"], "score": score,
                "delta": score - reference_score,
                "absolute_delta": abs(score - reference_score),
                "score_disagreement": abs(score - reference_score) > 1e-9,
                "fully_supported_decision_disagreement": (score == 1.0) != (reference_score == 1.0),
            })
        all_control.extend(groups_delta["unchanged"])
        all_changed.extend(groups_delta["reordered"])
        per_case.append({"id": case_id, "original_reference_score": reference_score,
                         "unchanged": delta_stats(groups_delta["unchanged"]),
                         "reordered": delta_stats(groups_delta["reordered"]),
                         "comparisons": groups_delta})
    return {
        "cases": len(groups), "trials": len(rows), "failed_trials": len(failures),
        "unchanged_variability": delta_stats(all_control),
        "reordered_sensitivity": delta_stats(all_changed), "per_case": per_case,
        "failures": [{"case_id": r["case_id"], "variant": r["variant"], "error": r.get("error") or "invalid judgement"} for r in failures],
        "interpretation": "Descriptive fixed-answer judge sensitivity. Reordering differences must be compared with unchanged-order variability; this small sample does not establish systematic position bias or generation-order effects.",
        "decision_threshold": "Fully supported iff faithfulness score equals 1.0; also report continuous score differences.",
    }


def delta_stats(rows: list[dict]) -> dict:
    n = len(rows)
    return {"valid_comparisons": n,
            "mean_signed_delta": sum(r["delta"] for r in rows) / n if n else None,
            "mean_absolute_delta": sum(r["absolute_delta"] for r in rows) / n if n else None,
            "score_disagreement_rate": sum(r["score_disagreement"] for r in rows) / n if n else None,
            "fully_supported_decision_disagreement_rate": sum(r["fully_supported_decision_disagreement"] for r in rows) / n if n else None}
