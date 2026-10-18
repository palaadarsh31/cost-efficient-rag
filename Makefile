PY ?= python

.PHONY: install ingest reingest query serve test eval eval-retrieval smoke bench cost corpus diagram

install:
	$(PY) -m pip install -r requirements-dev.txt

ingest:
	$(PY) -m docqa ingest data/corpus

reingest:  ## run twice: the second run should write 0 chunks
	$(PY) -m docqa ingest data/corpus

query:
	$(PY) -m docqa query "What is the maximum time a reviewer should take to respond to a code review?"

serve:
	$(PY) -m docqa serve --port 8000

test:
	$(PY) -m pytest

eval:
	$(PY) -m docqa.evaluation.run

eval-retrieval:  ## no LLM calls
	$(PY) -m docqa.evaluation.run --skip-answers

smoke:  ## whole harness with a hash embedder and a stub LLM, no network
	$(PY) -m docqa.evaluation.run --offline --repeats 1

bench:
	$(PY) scripts/bench_store.py --sizes 100000 1000000

cost:
	$(PY) scripts/cost_model.py

corpus:
	$(PY) scripts/build_corpus.py

diagram:
	dot -Tpng -Gdpi=150 docs/architecture.dot -o docs/architecture.png
	dot -Tsvg docs/architecture.dot -o docs/architecture.svg

# Problem 2: separate judge pipeline, standalone and independent of sqlite-vec
.PHONY: judge-validate judge-smoke judge-eval judge-tests
judge-validate:
	$(PY) -m judgepipe validate --suite judgepipe/data/suite.json
judge-smoke:
	$(PY) scripts/problem2_smoke.py
judge-tests:
	$(PY) -m pytest tests/test_judgepipe.py -q
judge-eval:
	$(PY) -m judgepipe run --suite judgepipe/data/suite.json --out results/problem2/live --repeats 2 --include-baseline
