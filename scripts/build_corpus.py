"""Rebuild data/corpus from its upstream sources (pinned commits).

The built corpus is committed, so you only need this to check provenance or to
regenerate it:

    python scripts/build_corpus.py               # clones the sources into a temp dir
    python scripts/build_corpus.py --sources DIR # reuse existing clones in DIR

Formats: Twelve-Factor and Prometheus pages stay Markdown, the two CNCF
whitepapers are the original PDFs, and Google's code review guide is rendered
to standalone HTML pages (header/nav/footer included) so the HTML loader gets
exercised on realistic markup.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus"

SOURCES = {
    "12factor": {
        "repo": "https://github.com/heroku/12factor",
        "commit": "1385d2c80bac38c25647651f6f5ec769561828dc",
        "license": "MIT",
    },
    "eng-practices": {
        "repo": "https://github.com/google/eng-practices",
        "commit": "3bb3ec25b3b0199f4940b1aa75f0ac5c5753301c",
        "license": "CC-BY 3.0",
    },
    "prometheus-docs": {
        "repo": "https://github.com/prometheus/docs",
        "commit": "e473448fce4e22011bb79167d690698f736876b1",
        "license": "Apache-2.0",
    },
    "tag-security": {
        "repo": "https://github.com/cncf/tag-security",
        "commit": "5fb87f474808e02e7786d7c5548e65ffc1a6e29e",
        "license": "Apache-2.0",
    },
}

TWELVE_FACTOR = [
    "codebase", "dependencies", "config", "backing-services", "build-release-run", "processes",
    "port-binding", "concurrency", "disposability", "dev-prod-parity", "logs", "admin-processes",
]
CODE_REVIEW = [
    "review/reviewer/standard.md", "review/reviewer/looking-for.md", "review/reviewer/navigate.md",
    "review/reviewer/speed.md", "review/reviewer/comments.md", "review/reviewer/pushback.md",
    "review/developer/cl-descriptions.md", "review/developer/small-cls.md",
    "review/developer/handling-comments.md", "review/emergencies.md",
]
PROMETHEUS = [
    "docs/concepts/data_model.md", "docs/concepts/jobs_instances.md", "docs/concepts/metric_types.md",
    "docs/practices/naming.md", "docs/practices/instrumentation.md", "docs/practices/histograms.md",
    "docs/practices/alerting.md", "docs/practices/rules.md", "docs/practices/pushing.md",
]
CNCF_PDFS = {
    "community/resources/security-whitepaper/v2/CNCF_cloud-native-security-whitepaper-May2022-v2.pdf":
        "cloud-native-security-whitepaper-v2.pdf",
    "community/working-groups/supply-chain-security/supply-chain-security-paper-v2/"
    "Software_Supply_Chain_Practices_whitepaper_v2.pdf":
        "software-supply-chain-best-practices-v2.pdf",
}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} | eng-practices</title>
<link rel="stylesheet" href="/eng-practices/assets/css/style.css">
<script>window.dataLayer = window.dataLayer || [];</script>
</head>
<body>
<header class="site-header">
  <nav><a href="/eng-practices/">Google Engineering Practices</a> /
  <a href="/eng-practices/review/">Code Review</a></nav>
</header>
<main id="content">
{body}
</main>
<footer class="site-footer">
  <p>Source: google/eng-practices ({path}), licensed under CC-BY 3.0.</p>
</footer>
</body>
</html>
"""


def fetch(name: str, dest: Path) -> Path:
    """Blobless fetch of the pinned commit; files are checked out one by one later."""
    meta = SOURCES[name]
    target = dest / name
    if (target / ".git").exists():
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if head != meta["commit"]:
            raise SystemExit(f"{target} is at {head}, expected {meta['commit']}")
        return target
    target.mkdir(parents=True)
    run = lambda *cmd: subprocess.run(["git", "-C", str(target), *cmd], check=True)
    run("init", "-q")
    run("remote", "add", "origin", meta["repo"])
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", meta["commit"])
    run("update-ref", "refs/heads/pinned", "FETCH_HEAD")
    run("symbolic-ref", "HEAD", "refs/heads/pinned")
    return target


def ensure(repo: Path, rel: str) -> Path:
    path = repo / rel
    if not path.exists():
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "HEAD", "--", rel], check=True)
    return path


def render_code_review(src: Path, rel: str) -> tuple[str, str]:
    text = ensure(src, rel).read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["extra", "sane_lists"])
    title = next((line.lstrip("# ").split("{")[0].strip() for line in text.splitlines()
                  if line.startswith("# ")), Path(rel).stem)
    return title, PAGE.format(title=title, body=body, path=rel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, help="directory with (or for) the source clones")
    args = parser.parse_args()

    workdir = args.sources or Path(tempfile.mkdtemp(prefix="corpus-src-"))
    src = {name: fetch(name, workdir) for name in SOURCES}

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("twelve-factor", "code-review", "prometheus", "cncf"):
        (OUT / sub).mkdir(parents=True)

    for page in TWELVE_FACTOR:
        shutil.copy(ensure(src["12factor"], f"content/en/{page}.md"), OUT / "twelve-factor" / f"{page}.md")
    for rel in CODE_REVIEW:
        _, html = render_code_review(src["eng-practices"], rel)
        name = rel.replace("review/", "").replace("/", "-").replace(".md", ".html")
        (OUT / "code-review" / name).write_text(html, encoding="utf-8")
    for rel in PROMETHEUS:
        name = Path(rel).parent.name + "-" + Path(rel).name.replace("_", "-")
        shutil.copy(ensure(src["prometheus-docs"], rel), OUT / "prometheus" / name)
    for rel, name in CNCF_PDFS.items():
        shutil.copy(ensure(src["tag-security"], rel), OUT / "cncf" / name)

    lines = ["# Corpus sources", "",
             "Built by `scripts/build_corpus.py`. All content is used under its upstream license.", "",
             "| Collection | Format | Upstream | Commit | License |", "|---|---|---|---|---|"]
    rows = [("twelve-factor", "Markdown", "12factor"), ("code-review", "HTML (rendered from Markdown)",
            "eng-practices"), ("prometheus", "Markdown", "prometheus-docs"), ("cncf", "PDF", "tag-security")]
    for collection, fmt, name in rows:
        meta = SOURCES[name]
        lines.append(f"| {collection} | {fmt} | {meta['repo']} | `{meta['commit'][:10]}` | {meta['license']} |")
    (ROOT / "data" / "CORPUS_SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {len(files)} files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
