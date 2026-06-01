"""Statistical testing for CTS evaluation (paper §7.1).

Paper: "5 seeds (3 full re-trainings + 2 inference-only);
95% CI via bootstrap (1000 resamples);
Wilcoxon signed-rank; Bonferroni-corrected for 12 primary comparisons
(alpha = 0.05/12)."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import random


@dataclass
class StatisticalResult:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    ci_level: float = 0.95


def bootstrap_ci(
    scores: List[float],
    *,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> StatisticalResult:
    """Bootstrap confidence interval (paper §7.1: 1000 resamples, 95% CI)."""
    n = len(scores)
    if n == 0:
        return StatisticalResult(0.0, 0.0, 0.0, 0.0, 0, ci_level)

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / len(sample))

    means.sort()
    alpha = 1.0 - ci_level
    lo_idx = int(math.floor(alpha / 2 * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2) * n_resamples)) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    mean_val = sum(scores) / n
    std_val = (sum((x - mean_val) ** 2 for x in scores) / max(n - 1, 1)) ** 0.5

    return StatisticalResult(
        mean=mean_val,
        std=std_val,
        ci_lower=means[lo_idx],
        ci_upper=means[hi_idx],
        n_samples=n,
        ci_level=ci_level,
    )


def wilcoxon_signed_rank(
    x: List[float],
    y: List[float],
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test (paper §7.1).

    Returns (W_statistic, approximate_p_value).
    Uses normal approximation for n >= 10.
    """
    n = min(len(x), len(y))
    if n == 0:
        return 0.0, 1.0

    diffs = [(x[i] - y[i]) for i in range(n)]
    diffs = [(d, i) for i, d in enumerate(diffs) if abs(d) > 1e-12]

    if not diffs:
        return 0.0, 1.0

    abs_diffs = sorted(diffs, key=lambda t: abs(t[0]))
    ranks = []
    i = 0
    while i < len(abs_diffs):
        j = i + 1
        while j < len(abs_diffs) and abs(abs(abs_diffs[j][0]) - abs(abs_diffs[i][0])) < 1e-12:
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            ranks.append((abs_diffs[k][0], avg_rank))
        i = j

    w_plus = sum(r for d, r in ranks if d > 0)
    w_minus = sum(r for d, r in ranks if d < 0)
    w = min(w_plus, w_minus)

    nr = len(ranks)
    if nr < 10:
        return w, 1.0

    mu = nr * (nr + 1) / 4.0
    sigma = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0)
    if sigma < 1e-12:
        return w, 1.0

    z = (w - mu) / sigma
    p = 2.0 * _normal_cdf(-abs(z))

    return w, p


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bonferroni_correct(p_values: List[float], n_comparisons: int = 12) -> List[float]:
    """Bonferroni correction (paper: alpha = 0.05/12)."""
    return [min(1.0, p * n_comparisons) for p in p_values]


def format_result(
    name: str,
    stat: StatisticalResult,
    *,
    pct: bool = True,
) -> str:
    """Format result as paper-style string: "64.1+-0.8%"."""
    mult = 100.0 if pct else 1.0
    ci_half = (stat.ci_upper - stat.ci_lower) / 2.0 * mult
    return f"{name}: {stat.mean * mult:.1f}+-{ci_half:.1f}{'%' if pct else ''}"


def multi_seed_aggregate(
    seed_results: Dict[int, List[float]],
    *,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
) -> StatisticalResult:
    """Aggregate results across seeds (paper: 3 re-train + 2 inference = 5+ runs)."""
    all_scores = []
    for scores in seed_results.values():
        all_scores.extend(scores)
    return bootstrap_ci(all_scores, n_resamples=n_resamples, ci_level=ci_level)
