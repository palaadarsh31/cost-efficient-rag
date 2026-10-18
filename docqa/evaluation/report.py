"""Render summary.json as a Markdown report."""
from __future__ import annotations


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(stat: dict | None, nd=3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    ci = stat.get("ci95")
    return f"{stat['mean']:.{nd}f}" + (f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]" if ci else "")


def render_markdown(s: dict) -> str:
    run, ing = s["run"], s["ingestion"]
    st = run["settings"]
    lines = ["# Evaluation results", ""]
    if run.get("offline"):
        lines += ["> **Offline smoke run** (hash embedder + stub LLM). These numbers only show the",
                  "> harness works end to end; they are not quality measurements.", ""]
    lines += [
        f"Generated {run['timestamp']} at commit `{run.get('git_commit')}` "
        f"(docqa {run['docqa_version']}, Python {run['python']}).",
        "",
        f"- Embeddings: `{run['embed_model']}` ({run['embed_dim']}-d), stored as float32 in sqlite-vec",
        f"- Chunking: {st['chunk_size']} tokens, {st['chunk_overlap']} overlap; k = {st['top_k']}; "
        f"default mode `{st['retrieval_mode']}`; similarity floor {st['min_score']}",
        f"- Hardware: {run['cpu']} ({run['cpu_count']} cores), "
        f"{str(run['ram_gb']) + ' GB RAM' if run['ram_gb'] else 'RAM n/a'}, {run['platform']}",
        "",
        "## Ingestion",
        "",
        "| | value |", "|---|---|",
        f"| documents (by type) | {ing['counts']['documents']} ({', '.join(f'{k}: {v}' for k, v in sorted(ing['documents_by_type'].items()))}) |",
        f"| chunks / vectors / FTS rows | {ing['counts']['chunks']} / {ing['counts']['vectors']} / {ing['counts']['fts_rows']} |",
        f"| chunk tokens mean (min-max) | {ing['chunk_tokens']['mean']:.0f} ({ing['chunk_tokens']['min']}-{ing['chunk_tokens']['max']}) |",
        f"| index size on disk | {ing['db_bytes'] / 1e6:.2f} MB |",
        f"| first ingest | {ing['first_run']['added']} added, {ing['first_run']['chunks_written']} chunks, {ing['first_run']['seconds']:.1f} s |",
        f"| re-ingest (same corpus) | {ing['second_run']['unchanged']} unchanged, {ing['second_run']['chunks_written']} chunks written, "
        f"vectors {ing['second_run']['vectors_before']} -> {ing['second_run']['vectors_after']}, {ing['second_run']['seconds']:.2f} s |",
        f"| idempotent | {'yes' if ing['idempotent'] else 'NO'} |",
        "",
        "## Retrieval",
        "",
        "Answerable questions only. A chunk is relevant if it contains at least half of a gold",
        "evidence quote; recall is over evidence facts. 95% bootstrap CIs in brackets.",
        "",
    ]
    for mode in ("dense", "hybrid"):
        r = s["retrieval"][mode]
        lines += [f"**{mode}** ({r['n_questions']} questions)", "",
                  "| k | Hit@k | Recall@k | MRR@k | nDCG@k | Precision@k | Context precision@k |",
                  "|---|---|---|---|---|---|---|"]
        for k, row in r["by_k"].items():
            lines.append(f"| {k} | " + " | ".join(_ci(row[n]) if str(k) == "5" else _f(row[n]["mean"])
                                               for n in ("hit", "recall", "mrr", "ndcg", "precision",
                                                         "context_precision")) + " |")
        lines.append("")

    audit = ing.get("embedding_input_audit", {})
    if audit.get("available"):
        lines += ["## Embedding tokenizer audit", "",
                  f"Actual model-token limit: {audit['max_length']}; truncated inputs: {audit['n_truncated']}/{audit['n_inputs']}; "
                  f"discarded tokens: {audit['tokens_discarded']}. Title/section headers and special tokens are included.", ""]
    if s.get("ablation_chunking"):
        lines += ["| Chunk size/overlap | Inputs truncated by embedding tokenizer |", "|---|---:|"]
        for row in s["ablation_chunking"]:
            audit = row.get("embedding_input_audit", {})
            value = f"{audit.get('n_truncated')}/{audit.get('n_inputs')}" if audit.get("available") else "Not measured"
            lines.append(f"| {row['chunk_size']}/{row['chunk_overlap']} | {value} |")
        lines.append("")

    lines += ["## Retrieval latency", "",
              "Query embedding + vector/keyword search + chunk fetch, measured in-process, "
              "warm, each question repeated.", "",
              "| mode | k | samples | embed p50 | search p50 | total p50 | total p95 | total max |",
              "|---|---|---|---|---|---|---|---|"]
    for mode, lat in s["latency_ms"].items():
        lines.append(f"| {mode} | {lat['k']} | {lat['total']['n']} | {_f(lat['embed']['p50'], 2)} | "
                     f"{_f(lat['search']['p50'], 2)} | {_f(lat['total']['p50'], 2)} | "
                     f"{_f(lat['total']['p95'], 2)} | {_f(lat['total']['max'], 2)} |")
    lines.append("")

    th = s["threshold"]
    lines += ["## Similarity floor (no-context gate)", "",
              f"Lowest top-1 similarity among answerable questions: {_f(th['answerable_min'])}; "
              f"highest among unanswerable: {_f(th['unanswerable_max'])}. Highest floor that keeps every "
              f"answerable question: {_f(th.get('highest_threshold_keeping_all_answerable'), 2)}.", "",
              "| threshold | answerable kept | unanswerable blocked |", "|---|---|---|"]
    for row in th["sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {_f(row['answerable_kept'], 2)} | {_f(row['unanswerable_blocked'], 2)} |")
    lines.append("")

    if s.get("ablation_chunking"):
        lines += ["## Chunking ablation", "",
                  "| size / overlap | chunks | mean tokens | dense Recall@5 | dense MRR@5 | dense nDCG@5 | "
                  "dense P@5 | hybrid Recall@5 | hybrid MRR@5 | index MB |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in s["ablation_chunking"]:
            d, h = row["dense@5"], row["hybrid@5"]
            lines.append(f"| {row['chunk_size']} / {row['chunk_overlap']} | {row['n_chunks']} | "
                         f"{row['mean_chunk_tokens']:.0f} | {_f(d['recall'])} | {_f(d['mrr'])} | {_f(d['ndcg'])} | "
                         f"{_f(d['precision'])} | {_f(h['recall'])} | {_f(h['mrr'])} | {row['db_bytes'] / 1e6:.2f} |")
        lines.append("")

    a = s.get("answers")
    if a:
        ab, cit, tok, lat = a["abstention"], a["citations"], a["tokens"], a["latency_ms"]
        lines += [
            "## Answer quality", "",
            f"Generator `{a['generator_model']}`, judge `{a['judge_model']}`. "
            f"Judge JSON parse failures: {a['judge_parse_errors']}.", "",
            f"Generation failures: {a.get('generation_failures', 0)}. Failures are not counted as correct abstentions.", "",
            "| metric | value [95% CI] | n | notes |", "|---|---|---|---|",
            f"| faithfulness | {_ci(a['faithfulness'])} | {a['faithfulness']['n']} | supported claims / claims, over all answered questions |",
            f"| answer relevance | {_ci(a['relevance'])} | {a['relevance']['n']} | judge 1-5 rescaled to 0-1 |",
            f"| correctness vs gold | {_ci(a['correctness'])} | {a['correctness']['n']} | judge: correct 1 / partial 0.5 / incorrect 0; {a['correctness_labels']} |",
            f"| exact match | {_ci(a['em'])} | {a['em']['n']} | SQuAD-style; punctuation replaced with spaces, citations stripped |",
            f"| token F1 | {_ci(a['f1'])} | {a['f1']['n']} | |",
            f"| contains gold | {_ci(a['contains_gold'])} | {a['contains_gold']['n']} | normalised gold string appears in answer |",
            "",
            "| abstention / citations | value |", "|---|---|",
            f"| unanswerable correctly declined | {ab['unanswerable_abstained']} / {ab['unanswerable_total']} |",
            f"| answerable wrongly declined | {ab['answerable_abstained']} / {ab['answerable_total']} |",
            f"| declined by similarity gate / by model | {ab['reasons']['no_relevant_context']} / {ab['reasons']['model_abstained']} |",
            f"| answers with >=1 valid citation | {cit['answered_with_citation']} / {cit['answered']} |",
            f"| invalid citation markers | {cit['invalid_citation_markers']} |",
            f"| gold-evidence citation precision | {_ci(cit['cited_chunk_precision'])} |",
            "", "Gold-evidence citation precision tests overlap with labeled answer evidence, not claim-to-citation entailment; extra supported claims can cite chunks outside that label set.",
            "",
            "| latency (ms) | p50 | p95 | n |", "|---|---|---|---|",
            f"| end-to-end, all questions | {_f(lat['end_to_end_all']['p50'], 1)} | {_f(lat['end_to_end_all']['p95'], 1)} | {lat['end_to_end_all']['n']} |",
            f"| end-to-end, LLM called | {_f(lat['end_to_end_llm_called']['p50'], 1)} | {_f(lat['end_to_end_llm_called']['p95'], 1)} | {lat['end_to_end_llm_called']['n']} |",
            f"| LLM call only | {_f(lat['llm']['p50'], 1)} | {_f(lat['llm']['p95'], 1)} | {lat['llm']['n']} |",
            "",
            f"Tokens per generation call: {_f(tok['prompt_mean'], 0)} prompt + {_f(tok['completion_mean'], 0)} completion "
            f"(totals {tok['prompt_total']} + {tok['completion_total']} over {tok['llm_calls']} calls"
            + (", estimated" if tok["estimated"] else "") + f"). Judge used {tok['judge_total']} tokens.",
        ]
        if a.get("llm_cost"):
            c = a["llm_cost"]
            if c.get('usd_per_1k_queries') is not None:
                lines.append(f"Generation cost at ${c['price_input_per_m']}/M input and ${c['price_output_per_m']}/M output: "
                             f"${c['usd_per_1k_queries']:.4f} per 1,000 user queries (including gated queries).")
            if c.get('usd_per_1k_generation_calls') is not None:
                lines.append(f"Per 1,000 actual generation calls: ${c['usd_per_1k_generation_calls']:.4f}.")
        lines.append("")
    else:
        lines += ["## Answer quality", "", "Not run (`--skip-answers`).", ""]

    lines.append(f"_Total harness time: {s['elapsed_seconds']:.1f} s._")
    return "\n".join(lines) + "\n"
