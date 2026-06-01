"""Unit tests for cts.eval.statistics (paper §7.1: 5 seeds, 95% bootstrap CI,
Wilcoxon signed-rank, Bonferroni correction).

These tests are reviewer-critical because the paper makes specific
statistical-rigor claims that depend on these helpers being correct.
We validate against (a) hand-computed values for trivial cases and
(b) scipy reference values within documented tolerance for the normal
approximation. scipy itself is *not* imported by the test (the
production code in `cts.eval.statistics` is intentionally pure-stdlib
to keep the dependency surface small for reviewers); the scipy values
are hard-coded constants computed once offline.
"""

from __future__ import annotations

import math

import pytest

from cts.eval.statistics import (
    StatisticalResult,
    bonferroni_correct,
    bootstrap_ci,
    format_result,
    multi_seed_aggregate,
    wilcoxon_signed_rank,
)


# ---------- bootstrap_ci ----------------------------------------------------

def test_bootstrap_ci_empty_input_returns_zero():
    res = bootstrap_ci([])
    assert isinstance(res, StatisticalResult)
    assert res.n_samples == 0
    assert res.mean == 0.0
    assert res.ci_lower == 0.0
    assert res.ci_upper == 0.0


def test_bootstrap_ci_constant_data_collapses_to_mean():
    # When every sample is the same, every resample is the same, so CI is degenerate.
    res = bootstrap_ci([0.5] * 20, n_resamples=200, seed=0)
    assert res.mean == pytest.approx(0.5)
    assert res.std == pytest.approx(0.0, abs=1e-9)
    assert res.ci_lower == pytest.approx(0.5)
    assert res.ci_upper == pytest.approx(0.5)
    assert res.n_samples == 20


def test_bootstrap_ci_mean_matches_sample_mean():
    scores = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]  # mean = 0.6
    res = bootstrap_ci(scores, n_resamples=500, seed=42)
    assert res.mean == pytest.approx(0.6)
    # bootstrap CI of a Bernoulli with mean 0.6 and n=10 should bracket 0.6
    assert res.ci_lower < 0.6 < res.ci_upper


def test_bootstrap_ci_is_deterministic_under_fixed_seed():
    scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    a = bootstrap_ci(scores, n_resamples=200, seed=1234)
    b = bootstrap_ci(scores, n_resamples=200, seed=1234)
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper


def test_bootstrap_ci_std_is_unbiased_sample_std():
    # std should be the n-1 (Bessel-corrected) sample std, not population std
    scores = [0.0, 0.0, 0.0, 0.0, 1.0]  # mean=0.2, sample std=sqrt(0.2)
    res = bootstrap_ci(scores, n_resamples=100)
    expected_std = math.sqrt(sum((x - 0.2) ** 2 for x in scores) / 4)
    assert res.std == pytest.approx(expected_std, rel=1e-6)


# ---------- wilcoxon_signed_rank --------------------------------------------

def test_wilcoxon_empty_input_returns_unity_p():
    w, p = wilcoxon_signed_rank([], [])
    assert w == 0.0
    assert p == 1.0


def test_wilcoxon_all_equal_returns_unity_p():
    w, p = wilcoxon_signed_rank([1.0] * 5, [1.0] * 5)
    assert w == 0.0
    assert p == 1.0


def test_wilcoxon_small_n_bails_out_to_unity():
    # The implementation explicitly returns p=1.0 for nr < 10 because the
    # normal approximation is unreliable below that. Reviewers reading the
    # paper should know this; lock it down with a test.
    x = [1.0, 0.0, 1.0, 0.0, 1.0]
    y = [0.0, 0.0, 0.0, 0.0, 0.0]
    _, p = wilcoxon_signed_rank(x, y)
    assert p == 1.0


def test_wilcoxon_large_diff_normal_approx_matches_scipy_within_tolerance():
    # Hand-picked 12-element pairs where x dominates y on every entry.
    # scipy.stats.wilcoxon(x, y, method='approx') gives p = 0.001925...;
    # our impl uses the same normal approximation but no continuity
    # correction, so we accept within +/- 30% of scipy's p.
    x = [0.55, 0.62, 0.45, 0.71, 0.58, 0.63, 0.49, 0.66, 0.52, 0.60, 0.57, 0.64]
    y = [0.42, 0.51, 0.39, 0.58, 0.44, 0.52, 0.38, 0.55, 0.41, 0.49, 0.46, 0.53]
    w, p = wilcoxon_signed_rank(x, y)
    assert w == 0.0  # x dominates y, so all signed ranks are positive => W- = 0
    scipy_p = 0.001925774646
    assert abs(p - scipy_p) / scipy_p < 0.30, f"got p={p}, scipy={scipy_p}"
    # Most importantly: stays clearly below alpha=0.05/12 = 0.00417
    assert p < 0.00417


def test_wilcoxon_no_difference_gives_high_p():
    # x and y interleaved roughly equally => non-significant
    x = [0.6, 0.55, 0.62, 0.58, 0.59, 0.61, 0.57, 0.60, 0.63, 0.56, 0.58, 0.62]
    y = [0.55, 0.58, 0.61, 0.59, 0.62, 0.56, 0.60, 0.57, 0.59, 0.61, 0.58, 0.60]
    _, p = wilcoxon_signed_rank(x, y)
    assert p > 0.05


# ---------- bonferroni_correct ----------------------------------------------

def test_bonferroni_multiplies_each_pvalue_by_n_comparisons():
    raw = [0.001, 0.01, 0.05, 0.1, 0.5]
    out = bonferroni_correct(raw, n_comparisons=12)
    expected = [min(1.0, p * 12) for p in raw]
    assert out == expected


def test_bonferroni_clamps_at_one():
    raw = [0.5, 0.9, 1.0]
    out = bonferroni_correct(raw, n_comparisons=12)
    for v in out:
        assert v <= 1.0


def test_bonferroni_default_n_comparisons_matches_paper():
    # Paper §7.1 says alpha = 0.05/12 (12 primary comparisons).
    raw = [0.004]  # just under alpha = 0.05/12 = 0.00417
    out = bonferroni_correct(raw)  # default n=12
    assert out[0] == pytest.approx(0.048, abs=1e-6)


# ---------- format_result ---------------------------------------------------

def test_format_result_paper_style():
    s = StatisticalResult(mean=0.641, std=0.012, ci_lower=0.633, ci_upper=0.649, n_samples=5)
    out = format_result("CTS-4nu", s, pct=True)
    assert "64.1" in out and "0.8" in out and "%" in out


def test_format_result_non_percentage():
    s = StatisticalResult(mean=2.0, std=0.5, ci_lower=1.5, ci_upper=2.5, n_samples=10)
    out = format_result("loss", s, pct=False)
    assert "%" not in out


# ---------- multi_seed_aggregate --------------------------------------------

def test_multi_seed_aggregate_concatenates_all_seeds():
    seed_results = {
        0: [0.6, 0.7, 0.5],
        1: [0.6, 0.6, 0.7],
        2: [0.5, 0.6, 0.6],
        3: [0.7, 0.7, 0.6],
        4: [0.6, 0.5, 0.7],
    }
    res = multi_seed_aggregate(seed_results, n_resamples=200)
    flat = [s for ss in seed_results.values() for s in ss]
    assert res.n_samples == len(flat)
    assert res.mean == pytest.approx(sum(flat) / len(flat))


def test_multi_seed_aggregate_handles_empty():
    res = multi_seed_aggregate({}, n_resamples=200)
    assert res.n_samples == 0
