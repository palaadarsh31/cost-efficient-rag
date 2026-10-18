from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings


def _setup_logging(settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _filters(args) -> dict:
    return {k: v for k, v in {"collection": args.collection, "doc_type": args.doc_type,
                              "doc_id": args.doc_id}.items() if v}


def _remove_index(db_path: Path) -> None:
    for p in db_path.parent.glob(db_path.name + "*"):
        p.unlink()


def cmd_ingest(args, settings: Settings) -> int:
    from .embeddings import build_embedder
    from .ingest import ingest_path
    from .store import VectorStore

    if args.rebuild:
        _remove_index(settings.db_path)
    embedder = build_embedder(settings)
    store = VectorStore(settings.db_path)
    report = ingest_path(args.path, store, embedder, settings, prune=args.prune)
    counts = store.counts()
    if args.json:
        print(json.dumps({**report.as_dict(), "index": counts}, indent=2))
    else:
        print(f"ingest {args.path} -> {settings.db_path}")
        print(f"  config: chunk_size={settings.chunk_size} overlap={settings.chunk_overlap} "
              f"model={embedder.model_name} dim={embedder.dim}")
        print(f"  files {report.files_seen} | added {report.added} | updated {report.updated} | "
              f"unchanged {report.unchanged} | removed {report.removed} | "
              f"chunks written {report.chunks_written}")
        print(f"  vectors before {report.vectors_before} -> after {report.vectors_after} | "
              f"documents {counts['documents']} | chunks {counts['chunks']} | "
              f"fts rows {counts['fts_rows']} | {report.seconds:.2f} s")
        for failure in report.failed:
            print(f"  FAILED {failure}")
    return 1 if report.failed else 0


def cmd_query(args, settings: Settings) -> int:
    from .qa import build_pipeline

    pipeline = build_pipeline(settings)
    result = pipeline.answer(args.question, k=args.k, filters=_filters(args), mode=args.mode)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print(f"\nQ: {result.question}\nA: {result.answer}\n")
    if result.citations:
        print("Sources:")
        for c in result.citations:
            where = c.doc_id + (f", p. {c.page}" if c.page else "") + (f", {c.section}" if c.section else "")
            sim = f"{c.similarity:.3f}" if c.similarity is not None else "-"
            print(f"  [{c.n}] {c.title} ({where}) sim={sim}")
    t, u = result.timings_ms, result.usage
    print("\n---")
    print(f"retrieval {t['retrieval']:.1f} ms (embed {t['retrieval_embed']:.1f}, search "
          f"{t['retrieval_search']:.1f}, fetch {t['retrieval_fetch']:.1f}) | llm {t['llm']:.1f} ms | "
          f"total {t['total']:.1f} ms")
    top = f"{result.top_similarity:.3f}" if result.top_similarity is not None else "n/a"
    print(f"chunks {result.chunk_count} | top similarity {top} | tokens {u['prompt_tokens']:,} prompt + "
          f"{u['completion_tokens']:,} completion = {u['total_tokens']:,}"
          + (" (estimated)" if u.get("estimated") else ""))
    if result.abstained:
        print(f"abstained: {result.abstain_reason}")
    return 0


def cmd_retrieve(args, settings: Settings) -> int:
    from .qa import build_pipeline

    pipeline = build_pipeline(settings, with_llm=False)
    r = pipeline.retriever.retrieve(args.question, k=args.k, filters=_filters(args), mode=args.mode)
    for h in r.hits:
        sim = f"{h.similarity:.3f}" if h.similarity is not None else "  -  "
        page = f" p.{h.chunk.page_start}" if h.chunk.page_start else ""
        print(f"{h.rank:>2}. sim={sim} {h.chunk.doc_id}{page} :: {h.chunk.text[:110]!r}")
    print(f"--- {r.mode} k={r.k} | {r.timings_ms}")
    return 0


def cmd_stats(args, settings: Settings) -> int:
    from .store import VectorStore

    store = VectorStore(settings.db_path)
    print(json.dumps({"db_path": str(settings.db_path), "size_bytes": store.size_bytes(),
                      "meta": store.meta(), "counts": store.counts()}, indent=2))
    return 0


def cmd_serve(args, settings: Settings) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docqa", description="Grounded QA over a document corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest a folder (or file) of PDF/HTML/MD")
    p.add_argument("path")
    p.add_argument("--prune", action="store_true", help="drop indexed docs whose files are gone")
    p.add_argument("--rebuild", action="store_true", help="delete the index and start over")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ingest)

    for name, func, help_text in (("query", cmd_query, "ask a question (retrieval + LLM)"),
                                  ("retrieve", cmd_retrieve, "show retrieved chunks only")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question")
        p.add_argument("-k", type=int, default=None)
        p.add_argument("--mode", choices=["dense", "hybrid"], default=None)
        p.add_argument("--collection")
        p.add_argument("--doc-type", choices=["pdf", "html", "md"])
        p.add_argument("--doc-id")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("stats", help="index statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    _setup_logging(settings)
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
