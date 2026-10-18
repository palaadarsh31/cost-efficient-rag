import math

import pytest

from docqa.evaluation import metrics as M

RANKED = ["a", "x", "b", "y", "z"]
RELEVANT = {"a", "b", "c"}


def test_hit_and_reciprocal_rank():
    assert M.hit_at_k(RANKED, RELEVANT, 1) == 1.0
    assert M.hit_at_k(["x", "y"], RELEVANT, 2) == 0.0
    assert M.reciprocal_rank(["x", "y", "b"], RELEVANT, 5) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(["x", "y", "b"], RELEVANT, 2) == 0.0


def test_recall_is_over_facts_not_chunks():
    # two facts; the first can be found in either "a" or "a2" (overlapping chunks)
    facts = [{"a", "a2"}, {"b"}]
    assert M.recall_at_k(["a", "a2", "q"], facts, 3) == 0.5
    assert M.recall_at_k(["a2", "q", "b"], facts, 3) == 1.0
    assert M.recall_at_k(["a2", "q", "b"], facts, 2) == 0.5


def test_ndcg_by_hand():
    # relevant at ranks 1 and 3; 3 relevant exist, so the ideal list has them at 1, 2, 3
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert M.ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k(["a", "b", "c"], RELEVANT, 3) == pytest.approx(1.0)
    assert M.ndcg_at_k(["x"], RELEVANT, 1) == 0.0


def test_precision_and_context_precision():
    assert M.precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    # precision@1 = 1, precision@3 = 2/3, averaged over the two relevant positions
    assert M.context_precision_at_k(RANKED, RELEVANT, 5) == pytest.approx((1 + 2 / 3) / 2)
    assert M.context_precision_at_k(["x", "a"], RELEVANT, 2) == pytest.approx(0.5)
    assert M.context_precision_at_k(["x", "y"], RELEVANT, 2) == 0.0


def test_squad_normalisation():
    assert M.normalize_answer("The answer is: SIGTERM! [1]") == "answer is sigterm"
    assert M.exact_match("SIGTERM [2]", "sigterm") == 1.0
    assert M.exact_match("It is SIGTERM", "SIGTERM") == 0.0
    assert M.contains_gold("It is SIGTERM [1].", "SIGTERM") == 1.0


def test_token_f1():
    assert M.token_f1("build release and run", "build, release, and run") == 1.0
    p, r = 2 / 4, 2 / 3
    assert M.token_f1("one business day maybe", "one business week") == pytest.approx(2 * p * r / (p + r))
    assert M.token_f1("", "x") == 0.0


def test_percentile_and_bootstrap():
    vals = list(range(1, 101))
    assert M.percentile(vals, 50) == pytest.approx(50.5)
    assert M.percentile(vals, 95) == pytest.approx(95.05)
    lo, hi = M.bootstrap_ci([0.0, 1.0] * 20)
    assert lo < 0.5 < hi
    assert M.bootstrap_ci([1.0]) is None
