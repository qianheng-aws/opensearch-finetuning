from __future__ import annotations

import math

import pytest

from eval_metrics import ndcg_at_k_graded, paired_bootstrap_ci


def test_ndcg_perfect_ranking_is_one():
    ranking = ["d_a", "d_b", "d_c"]
    scores = {"d_a": 1.0, "d_b": 0.8, "d_c": 0.5}
    assert ndcg_at_k_graded(ranking, scores, k=3) == pytest.approx(1.0)


def test_ndcg_zero_relevant_returns_zero():
    assert ndcg_at_k_graded(["d_a"], {"d_a": 0.0}, k=1) == 0.0


def test_ndcg_imperfect_ranking_below_one():
    ranking = ["d_b", "d_a", "d_c"]
    scores = {"d_a": 1.0, "d_b": 0.5, "d_c": 0.0}
    val = ndcg_at_k_graded(ranking, scores, k=3)
    assert 0.0 < val < 1.0


def test_ndcg_missing_doc_treated_as_zero():
    ranking = ["d_x", "d_a"]
    scores = {"d_a": 1.0}
    expected_dcg = 0.0 + 1.0 / math.log2(3)
    expected_idcg = 1.0
    assert ndcg_at_k_graded(ranking, scores, k=2) == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_truncates_to_k_for_both_dcg_and_idcg():
    ranking = ["d_a", "d_b", "d_c", "d_d"]
    scores = {"d_a": 0.3, "d_b": 0.6, "d_c": 0.9, "d_d": 1.0}
    val = ndcg_at_k_graded(ranking, scores, k=2)
    expected_dcg = 0.3 + 0.6 / math.log2(3)
    expected_idcg = 1.0 + 0.9 / math.log2(3)
    assert val == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_empty_ranking_returns_zero():
    assert ndcg_at_k_graded([], {"d_a": 1.0}, k=10) == 0.0


def test_paired_bootstrap_constant_uplift():
    base = [0.4, 0.5, 0.6, 0.7, 0.8] * 20
    ft = [b + 0.1 for b in base]
    r = paired_bootstrap_ci(base, ft, num_resamples=500, seed=42)
    assert r.mean_diff == pytest.approx(0.1, abs=1e-9)
    assert r.ci_low == pytest.approx(0.1, abs=1e-9)
    assert r.ci_high == pytest.approx(0.1, abs=1e-9)


def test_paired_bootstrap_zero_diff_includes_zero():
    r = paired_bootstrap_ci([0.5] * 50, [0.5] * 50, num_resamples=500, seed=42)
    assert r.mean_diff == 0.0
    assert r.ci_low <= 0.0 <= r.ci_high


def test_paired_bootstrap_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([0.1, 0.2], [0.1], num_resamples=10, seed=1)


def test_paired_bootstrap_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_ci([], [], num_resamples=10, seed=1)
