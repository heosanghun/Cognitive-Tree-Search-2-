"""Shared helpers for paper Tables 13 / 15 sweep automation
(K = MCTS top-K children sensitivity, W = per-step simulation budget,
λ_halt training-reward ablation).

Every helper here is intentionally CPU-only and dependency-light so the
CI sweep dry-runs (and the ``tests/test_sweep_K_W_lambda.py`` regression
suite) execute in <1 s on a laptop.

The CONFIDENCE-INTERVAL primitive is INTENTIONALLY a tuple-returning
helper rather than the ``StatisticalResult`` dataclass used by
``cts/eval/statistics.py::bootstrap_ci``: the sweep aggregator needs a
3-tuple ``(mean, lo, hi)`` and the existing dataclass conflates that
with ``std`` / ``ci_level`` metadata which the sweep table does not
need. Both implementations agree on the underlying percentile-bootstrap
estimator (paper §7.1: 1000 resamples, α=0.05).
"""

from __future__ import annotations

import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


__all__ = [
    "bootstrap_ci",
    "render_sweep_markdown",
    "load_sweep_jsonl",
    "dry_run_grid",
    "append_sweep_row",
    "summarize_sweep",
]


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    *,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    """Percentile bootstrap mean confidence interval.

    Returns ``(mean, lo, hi)``. Constant input (or single-sample input)
    collapses to ``lo == hi == mean`` by construction — pinned by
    ``test_bootstrap_ci_constant_input``.

    ``alpha = 0.05`` produces a 95% two-sided interval (paper §7.1
    headline statistical protocol). ``n_resamples = 1000`` matches the
    paper's resample count.

    Empty input returns ``(0.0, 0.0, 0.0)`` rather than raising so the
    sweep aggregator can render partially-completed rows without a
    branch.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = sum(vals) / n
    if n == 1:
        return mean_val, mean_val, mean_val
    if all(abs(v - vals[0]) < 1e-12 for v in vals):
        return mean_val, mean_val, mean_val

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _i in range(n):
            s += vals[rng.randint(0, n - 1)]
        means.append(s / n)
    means.sort()

    lo_idx = int(math.floor(alpha / 2.0 * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return mean_val, means[lo_idx], means[hi_idx]


def load_sweep_jsonl(path: Union[Path, str]) -> List[Dict[str, Any]]:
    """Read a JSONL sweep results file into a list of dicts.

    Returns ``[]`` for missing / empty files so callers can treat
    "first-ever invocation" and "resumed re-run" identically. Bad
    individual lines are skipped (so a partially-flushed run from a
    Ctrl-C does not poison subsequent --resume invocations).
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_sweep_row(path: Union[Path, str], row: Dict[str, Any]) -> None:
    """Atomically append one JSONL row. Creates parent dir as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def dry_run_grid(
    param_values: Iterable[Any],
    seeds: Iterable[int],
    benchmarks: Iterable[str],
) -> List[Tuple[Any, int, str]]:
    """Cartesian product of ``(param, seed, benchmark)`` planning tuples.

    Used by every sweep script's ``--dry-run`` path so reviewers can
    audit the planned grid (and CI can exercise the full launch
    machinery) without spinning up a single GPU job.
    """
    params = list(param_values)
    seed_list = [int(s) for s in seeds]
    bench_list = [str(b) for b in benchmarks]
    return [(p, s, b) for p, s, b in product(params, seed_list, bench_list)]


def summarize_sweep(
    rows: Sequence[Dict[str, Any]],
    param_name: str,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Group per-(param, seed, problem) rows by ``param`` and compute
    bootstrap CI over the per-(param, seed) accuracy means.

    Each returned row has keys: ``param_value``, ``n_problems`` (total
    rows for that param value across seeds), ``n_seeds``, ``mean_acc``,
    ``ci_lo``, ``ci_hi``.

    The aggregation strategy is "per-seed accuracy first, then bootstrap
    over seeds" which matches paper §7.1's protocol for multi-seed
    evaluation reporting.
    """
    by_param: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        if param_name not in r:
            continue
        by_param.setdefault(r[param_name], []).append(r)

    out: List[Dict[str, Any]] = []
    for pv in sorted(by_param.keys(), key=lambda x: (str(type(x).__name__), x)):
        prs = by_param[pv]
        # Group by seed, take per-seed mean accuracy.
        per_seed: Dict[int, List[float]] = {}
        for r in prs:
            s = int(r.get("seed", 0))
            score = r.get("score")
            if score is None:
                score = r.get("accuracy")
            if score is None:
                continue
            per_seed.setdefault(s, []).append(float(score))

        seed_means = [sum(v) / len(v) for v in per_seed.values() if v]
        mean_acc, lo, hi = bootstrap_ci(
            seed_means, n_resamples=n_resamples, alpha=alpha,
        )
        out.append({
            "param_value": pv,
            "n_problems": len(prs),
            "n_seeds": len(per_seed),
            "mean_acc": mean_acc,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return out


def render_sweep_markdown(
    rows: List[Dict[str, Any]],
    param_name: str,
    out: Union[Path, str],
    *,
    title: Optional[str] = None,
) -> None:
    """Render a sweep summary table to a Markdown file.

    Each input row must have at minimum: ``param_value``, ``n_problems``,
    ``mean_acc``, ``ci_lo``, ``ci_hi``. Optional ``n_seeds`` is rendered
    when present.

    The resulting Markdown contains the literal ``param_name`` and the
    canonical ``mean ± 95% CI`` header so downstream regression tests
    can grep for both (pinned by
    ``test_render_sweep_markdown_writes_table``).
    """
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Sweep results: {param_name}")
    lines.append("")
    if title:
        lines.append(title)
        lines.append("")
    has_seeds = any("n_seeds" in r for r in rows)
    header_cells = [param_name, "n_problems"]
    if has_seeds:
        header_cells.append("n_seeds")
    header_cells.extend(["mean_acc", "ci_lo", "ci_hi", "mean ± 95% CI"])
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cells) + " |")
    for r in rows:
        pv = r.get("param_value")
        n = int(r.get("n_problems", 0))
        mean = float(r.get("mean_acc", 0.0))
        lo = float(r.get("ci_lo", mean))
        hi = float(r.get("ci_hi", mean))
        ci_half = (hi - lo) / 2.0
        cells = [str(pv), str(n)]
        if has_seeds:
            cells.append(str(int(r.get("n_seeds", 0))))
        cells.extend([
            f"{mean:.4f}",
            f"{lo:.4f}",
            f"{hi:.4f}",
            f"{mean*100:.1f} ± {ci_half*100:.1f}%",
        ])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
