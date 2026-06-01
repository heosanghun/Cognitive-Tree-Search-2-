"""Honest Hybrid-KV (paper §7.7) measurement infrastructure.

This module is **paper-faithful and disclosure-honest**: it measures only
what the current local code path actually executes — the per-leaf decision
overhead of ``hybrid_transition_decision`` and the cached-node statistics
exposed by ``HybridKVManager.report()`` — and provides a TOST equivalence
scaffold reviewers can run **once the KV-reuse fast path is wired into
``GemmaCTSBackbone``** (post-submission work tracked in
``cts/eval/cuda_graph_skeleton.py`` and the TODO inside
``cts/mcts/hybrid_kv.py::HybridKVManager.__init__``).

We do **NOT** pretend to measure the cache-HIT path that does not exist;
the README "Implementation Status" row already discloses

    "decision-plumbed; KV-reuse pending ... the −21% wall-clock figure
     therefore remains the paper's reference number, not a measured local
     result."

and this module stays consistent with that disclosure.

Public surface
--------------
* :func:`measure_decision_overhead` — long-form ``pandas.DataFrame``
  across ``(seed, problem_id, mode)`` cells. ``mode`` is one of
  :data:`ALL_MODES` (``"hybrid_off"`` / ``"hybrid_decision_only"``).
* :func:`tost_equivalence` — Schuirmann (1987) two one-sided t-tests
  for equivalence with margin ±``delta`` at level ``alpha``. SciPy is
  used when available; a regularised-incomplete-beta closed form is
  the fallback for scipy-less forks.
* :func:`summarize_hybrid_kv` — per-mode mean ± std + the TOST verdict
  between the two modes (margin = 5 % of the ``hybrid_off`` mean).
* :func:`render_hybrid_kv_markdown` — writes a Markdown report whose
  first 30 lines carry the literal phrase
  ``KV-reuse hit path NOT YET measured`` so reviewers cannot misread
  the report as a HIT-path measurement.
* :func:`write_trace_jsonl` — long-form per-cell JSONL persistence used
  by ``scripts/measure_hybrid_kv.py``.

CPU-only at import time. ``scipy.stats.t`` is imported lazily inside
:func:`_t_cdf` with a stdlib fallback.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch

from cts.backbone.protocol import BaseCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.mcts.cts_episode import CtsEpisodeResult, cts_full_episode
from cts.mcts.hybrid_kv import HybridKVManager
from cts.policy.meta_policy import MetaPolicy

# ---------------------------------------------------------------------------
# Public constants — pinned by ``tests/test_hybrid_kv_measurement.py``.
# ---------------------------------------------------------------------------

#: Mode label for the baseline run (no HybridKVManager attached).
MODE_OFF: str = "hybrid_off"

#: Mode label for the run that threads a HybridKVManager. The cache-HIT
#: path stays inert today (see the §7.7 TODO inside
#: ``HybridKVManager.__init__``); only the *decision overhead* and the
#: cached-node *statistics* are honestly measured.
MODE_DECISION_ONLY: str = "hybrid_decision_only"

#: All modes the harness can honestly measure today. The hypothetical
#: ``"hybrid_full"`` mode is deliberately omitted: we cannot accidentally
#: fabricate a measurement of code that does not exist.
ALL_MODES: Tuple[str, str] = (MODE_OFF, MODE_DECISION_ONLY)

#: Long-form DataFrame schema returned by
#: :func:`measure_decision_overhead`.
LONG_FORM_COLUMNS: List[str] = [
    "seed",
    "problem_id",
    "mode",
    "wall_seconds",
    "decision_calls",
    "cached_nodes",
    "vram_used_gb",
]

#: Top-of-report caveat. The literal substring
#: ``"KV-reuse hit path NOT YET measured"`` is asserted by
#: ``test_render_hybrid_kv_markdown_includes_caveat_at_top`` and copied
#: verbatim into the rendered Markdown's first 30 lines so reviewers
#: see the disclosure BEFORE any number.
KV_REUSE_CAVEAT: str = (
    "**KV-reuse hit path NOT YET measured.** The paper's −21 % wall-clock "
    "figure (§7.7) requires backbone-level `past_key_values` serialization "
    "that is not yet plumbed into `GemmaCTSBackbone`. This report measures "
    "only what the local pipeline can honestly observe today: (a) the "
    "decision overhead of consulting `HybridKVManager` on every leaf, and "
    "(b) the cached-node statistics surfaced by `HybridKVManager.report()`. "
    "The −21 % figure remains the **paper's reference number**, not a "
    "measured local result."
)


# ---------------------------------------------------------------------------
# 1. TOST equivalence test (Schuirmann 1987)
# ---------------------------------------------------------------------------

def _t_cdf(t: float, df: float) -> float:
    """CDF of Student's t. Uses ``scipy.stats.t`` when available; falls
    back to a regularised incomplete beta closed form otherwise. Both
    paths are accurate to ~1e-6 for ``df >= 2`` (TOST p-values only
    need ~1e-3, so this is overkill in the right direction)."""
    try:  # pragma: no cover - exercised on every developer machine
        from scipy import stats as _stats  # type: ignore
        return float(_stats.t.cdf(t, df))
    except Exception:  # pragma: no cover - scipy-stripped forks
        pass

    if df <= 0:
        return 0.5
    # Closed form: F(t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2) for t >= 0,
    # F(-t) = 1 - F(t) for t < 0.
    x = df / (df + t * t)
    ib = _regularised_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta - math.log(a))
    return front * _beta_cf(x, a, b)


def _beta_cf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Lentz's algorithm for the beta continued fraction."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def tost_equivalence(
    a: Sequence[float],
    b: Sequence[float],
    delta: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two one-sided t-tests for equivalence with margin ±``delta``
    (Schuirmann 1987).

    Tests the null pair:

    * H01: ``mean(a) - mean(b) <= -delta`` (reject ⇒ diff is not too low)
    * H02: ``mean(a) - mean(b) >= +delta`` (reject ⇒ diff is not too high)

    Equivalence is declared (``equivalent=True``) iff both nulls are
    rejected at level ``alpha``, i.e. ``max(p_lower, p_upper) < alpha``.

    Returns:
        Dict with ``p_lower``, ``p_upper``, ``p_max``, ``equivalent``
        (bool), ``n_a``, ``n_b``, ``mean_diff``, ``delta``.

    Edge cases:
      * Zero variance + ``|mean_diff| < delta`` → ``equivalent=True`` with
        all p-values 0.0 (``test_tost_equivalence_identical_samples``).
      * Zero variance + ``|mean_diff| >= delta`` → ``equivalent=False``
        with all p-values 1.0 (``test_tost_equivalence_far_apart``).
      * Empty input → ``equivalent=False`` with NaN-equivalent sentinels.
    """
    a_list = [float(x) for x in a]
    b_list = [float(x) for x in b]
    n_a, n_b = len(a_list), len(b_list)
    delta = float(delta)
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative (got {delta})")

    if n_a == 0 or n_b == 0:
        return {
            "p_lower": 1.0, "p_upper": 1.0, "p_max": 1.0,
            "equivalent": False,
            "n_a": int(n_a), "n_b": int(n_b),
            "mean_diff": 0.0, "delta": delta,
        }

    mean_a = sum(a_list) / n_a
    mean_b = sum(b_list) / n_b
    mean_diff = mean_a - mean_b

    var_a = (
        sum((x - mean_a) ** 2 for x in a_list) / max(n_a - 1, 1)
        if n_a > 1 else 0.0
    )
    var_b = (
        sum((x - mean_b) ** 2 for x in b_list) / max(n_b - 1, 1)
        if n_b > 1 else 0.0
    )
    se = math.sqrt(var_a / n_a + var_b / n_b)

    # Degenerate-variance fast path. Pinned by the ``identical`` and
    # ``far_apart`` tests.
    if se < 1e-15:
        equivalent = abs(mean_diff) < delta
        p = 0.0 if equivalent else 1.0
        return {
            "p_lower": p, "p_upper": p, "p_max": p,
            "equivalent": equivalent,
            "n_a": int(n_a), "n_b": int(n_b),
            "mean_diff": float(mean_diff), "delta": delta,
        }

    # Welch-Satterthwaite degrees of freedom.
    num = (var_a / n_a + var_b / n_b) ** 2
    denom_a = (var_a / n_a) ** 2 / max(n_a - 1, 1) if n_a > 1 else 0.0
    denom_b = (var_b / n_b) ** 2 / max(n_b - 1, 1) if n_b > 1 else 0.0
    denom = denom_a + denom_b
    df = num / denom if denom > 0 else float(max(n_a + n_b - 2, 1))
    df = max(df, 1.0)

    # Standard Schuirmann TOST.
    # H01: diff <= -delta → t_lower = (diff + delta)/se
    #   reject H01 when t_lower is large positive ⇒ p_lower = 1 - F_t(t_lower).
    # H02: diff >= +delta → t_upper = (diff - delta)/se
    #   reject H02 when t_upper is very negative ⇒ p_upper = F_t(t_upper).
    t_lower = (mean_diff + delta) / se
    t_upper = (mean_diff - delta) / se
    p_lower = 1.0 - _t_cdf(t_lower, df)
    p_upper = _t_cdf(t_upper, df)
    p_max = max(p_lower, p_upper)

    return {
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_max": float(p_max),
        "equivalent": bool(p_max < alpha),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_diff": float(mean_diff),
        "delta": float(delta),
    }


# ---------------------------------------------------------------------------
# 2. Decision-overhead measurement (CPU-friendly mock-backbone path)
# ---------------------------------------------------------------------------

@dataclass
class _MeasurementRow:
    seed: int
    problem_id: int
    mode: str
    wall_seconds: float
    decision_calls: int
    cached_nodes: int
    vram_used_gb: float


def _default_kv_manager_factory() -> HybridKVManager:
    """Default factory for the ``hybrid_decision_only`` mode (paper §7.7
    defaults: shallow_depth_limit=5, max_kv_vram_gb=1.0). A fresh
    instance per cell is required because the manager carries
    per-episode counters."""
    return HybridKVManager(shallow_depth_limit=5, max_kv_vram_gb=1.0)


def _run_one_episode(
    backbone: BaseCTSBackbone,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    *,
    prompt: str,
    seed: int,
    hybrid_kv_manager: Optional[HybridKVManager],
    K: int = 4,
    W: int = 2,
    tau_budget: float = 1e12,
    broyden_max_iter: int = 2,
    wall_clock_budget_s: float = 5.0,
    max_decode_tokens: int = 4,
) -> Tuple[float, CtsEpisodeResult]:
    """Run one CTS episode and return ``(wall_seconds, result)``."""
    t0 = time.perf_counter()
    res = cts_full_episode(
        prompt,
        backbone=backbone,
        meta_policy=meta_policy,
        critic=critic,
        W=W,
        K=K,
        tau_budget=tau_budget,
        broyden_max_iter=broyden_max_iter,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        top_k=2,
        max_decode_tokens=max_decode_tokens,
        device=torch.device("cpu"),
        wall_clock_budget_s=wall_clock_budget_s,
        z0_seed=int(seed),
        selection_seed=int(seed),
        hybrid_kv_manager=hybrid_kv_manager,
    )
    return time.perf_counter() - t0, res


def measure_decision_overhead(
    backbone: BaseCTSBackbone,
    problems: Sequence[Union[str, Dict[str, Any]]],
    n_seeds: int = 3,
    hybrid_kv_manager_factory: Optional[Callable[[], HybridKVManager]] = None,
    *,
    meta_policy: Optional[MetaPolicy] = None,
    critic: Optional[NeuroCritic] = None,
    K: int = 4,
    W: int = 2,
    tau_budget: float = 1e12,
    broyden_max_iter: int = 2,
    wall_clock_budget_s: float = 5.0,
    max_decode_tokens: int = 4,
    seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Measure §7.7 decision overhead and cache statistics, long-form.

    Returns a long-form ``pandas.DataFrame`` with columns
    ``[seed, problem_id, mode, wall_seconds, decision_calls,
    cached_nodes, vram_used_gb]`` where:

      * ``mode == "hybrid_off"``: baseline pure-DEQ episode loop with
        ``hybrid_kv_manager=None``. ``decision_calls`` / ``cached_nodes``
        / ``vram_used_gb`` are all 0 by construction.
      * ``mode == "hybrid_decision_only"``: same loop, but a fresh
        :class:`HybridKVManager` is threaded through every leaf. The
        HIT path stays inert today, so this mode measures only the
        *decision overhead* of having the manager wired in.

    Args:
        backbone: anything that satisfies
            :class:`cts.backbone.protocol.BaseCTSBackbone` — the
            CPU-friendly mock at ``cts/backbone/mock_tiny.py`` is the
            canonical CPU-only choice used by the test suite.
        problems: list of prompt strings, OR list of dicts with a
            ``"problem"`` (or ``"prompt"`` / ``"question"``) key — the
            AIME JSONL schema is supported transparently.
        n_seeds: number of seeds to sweep when ``seeds`` is None
            (defaults to ``range(n_seeds)``).
        hybrid_kv_manager_factory: zero-arg callable returning a fresh
            :class:`HybridKVManager`. Defaults to
            :func:`_default_kv_manager_factory`.
        meta_policy / critic: optional pre-built components. When None,
            fresh instances are built per the backbone hidden size.
        seeds: explicit list overriding ``n_seeds``.

    Notes:
        Wall-clock numbers from a CPU mock backbone are inherently noisy
        (millisecond-scale episodes); the mode comparison is most
        informative as an *upper bound* on the decision overhead, which
        is exactly what we want to declare equivalent via TOST. On a
        real GPU backbone this same harness becomes a real speedup
        measurement once the cache-HIT path is plumbed.
    """
    if hybrid_kv_manager_factory is None:
        hybrid_kv_manager_factory = _default_kv_manager_factory

    if meta_policy is None or critic is None:
        d = int(getattr(backbone, "hidden_size", 32))
        if meta_policy is None:
            meta_policy = MetaPolicy(text_dim=d, hidden=64, W=W)
        if critic is None:
            critic = NeuroCritic(z_dim=d)

    seed_iter: List[int] = (
        [int(s) for s in seeds] if seeds is not None else list(range(int(n_seeds)))
    )

    rows: List[_MeasurementRow] = []
    for seed in seed_iter:
        for problem_id, item in enumerate(problems):
            if isinstance(item, dict):
                prompt = (
                    item.get("problem")
                    or item.get("prompt")
                    or item.get("question")
                    or item.get("text")
                    or ""
                )
                prompt = str(prompt)
            else:
                prompt = str(item)

            for mode in ALL_MODES:
                kv = hybrid_kv_manager_factory() if mode == MODE_DECISION_ONLY else None
                wall_s, res = _run_one_episode(
                    backbone, meta_policy, critic,
                    prompt=prompt,
                    seed=seed,
                    hybrid_kv_manager=kv,
                    K=K, W=W,
                    tau_budget=tau_budget,
                    broyden_max_iter=broyden_max_iter,
                    wall_clock_budget_s=wall_clock_budget_s,
                    max_decode_tokens=max_decode_tokens,
                )

                if kv is not None:
                    rep = res.stats.get("hybrid_kv", {})
                    decision_calls = int(rep.get("decision_calls", 0))
                    cached_nodes = int(rep.get("cached_nodes", 0))
                    vram_used_gb = float(rep.get("vram_used_gb", 0.0))
                else:
                    decision_calls = 0
                    cached_nodes = 0
                    vram_used_gb = 0.0

                rows.append(_MeasurementRow(
                    seed=int(seed),
                    problem_id=int(problem_id),
                    mode=mode,
                    wall_seconds=float(wall_s),
                    decision_calls=decision_calls,
                    cached_nodes=cached_nodes,
                    vram_used_gb=vram_used_gb,
                ))

    return pd.DataFrame([r.__dict__ for r in rows], columns=LONG_FORM_COLUMNS)


# ---------------------------------------------------------------------------
# 3. Summary aggregator (mean ± std + TOST verdict)
# ---------------------------------------------------------------------------

def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(max(var, 0.0))


def summarize_hybrid_kv(
    df: pd.DataFrame,
    *,
    margin_frac: float = 0.05,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Aggregate a long-form measurement DataFrame into a summary dict.

    Returns:
        dict with::

            {
              "n_seeds":      int,
              "n_problems":   int,
              "margin_frac":  float,
              "by_mode": {
                MODE_OFF: {n, wall_seconds_mean, wall_seconds_std,
                           decision_calls_mean, cached_nodes_mean,
                           vram_used_gb_mean},
                MODE_DECISION_ONLY: {... same fields ...},
              },
              "tost":   {... :func:`tost_equivalence` output ...},
              "caveat": str (contains "KV-reuse hit path NOT YET measured"),
            }

    The TOST equivalence margin defaults to **5 % of the ``hybrid_off``
    mean wall-time** (paper §7.7 default).
    """
    summary: Dict[str, Any] = {
        "by_mode": {},
        "tost": {},
        "n_seeds": 0,
        "n_problems": 0,
        "margin_frac": float(margin_frac),
        "alpha": float(alpha),
        "caveat": KV_REUSE_CAVEAT,
    }

    if df is None or len(df) == 0:
        summary["tost"] = tost_equivalence([], [], delta=0.0, alpha=alpha)
        return summary

    summary["n_seeds"] = int(df["seed"].nunique())
    summary["n_problems"] = int(df["problem_id"].nunique())

    by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in ALL_MODES:
        sub = df[df["mode"] == mode]
        wall = [float(v) for v in sub["wall_seconds"].tolist()]
        mean_w, std_w = _mean_std(wall)
        by_mode[mode] = {
            "n": len(wall),
            "wall_seconds_mean": mean_w,
            "wall_seconds_std": std_w,
            "decision_calls_mean": (
                float(sub["decision_calls"].mean()) if len(sub) else 0.0
            ),
            "cached_nodes_mean": (
                float(sub["cached_nodes"].mean()) if len(sub) else 0.0
            ),
            "vram_used_gb_mean": (
                float(sub["vram_used_gb"].mean()) if len(sub) else 0.0
            ),
        }
    summary["by_mode"] = by_mode

    off_walls = [float(v) for v in df[df["mode"] == MODE_OFF]["wall_seconds"].tolist()]
    on_walls = [float(v) for v in df[df["mode"] == MODE_DECISION_ONLY]["wall_seconds"].tolist()]
    off_mean = by_mode[MODE_OFF]["wall_seconds_mean"]
    delta = max(margin_frac * abs(off_mean), 1e-9)
    summary["tost"] = tost_equivalence(off_walls, on_walls, delta=delta, alpha=alpha)
    return summary


# ---------------------------------------------------------------------------
# 4. Markdown rendering (caveat-first)
# ---------------------------------------------------------------------------

def _wrap_paragraph(text: str, width: int = 80) -> List[str]:
    """Cheap line-wrap so the rendered Markdown stays under ~80 cols.
    Avoids the ``textwrap`` dependency for the same stdlib-light reason
    the rest of this module uses ``math``."""
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for word in text.split():
        if cur_len + len(word) + (1 if cur else 0) > width and cur:
            out.append(" ".join(cur))
            cur = [word]
            cur_len = len(word)
        else:
            cur.append(word)
            cur_len += len(word) + (1 if len(cur) > 1 else 0)
    if cur:
        out.append(" ".join(cur))
    return out


def render_hybrid_kv_markdown(
    summary: Dict[str, Any],
    out_path: Union[Path, str],
) -> None:
    """Write a Markdown report whose first 30 lines carry the
    ``KV-reuse hit path NOT YET measured`` disclosure (asserted by
    ``tests/test_hybrid_kv_measurement.py::
    test_render_hybrid_kv_markdown_includes_caveat_at_top``).

    The output file is overwritten on every call (idempotent).
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    by_mode = summary.get("by_mode", {})
    tost = summary.get("tost", {})
    margin_frac = float(summary.get("margin_frac", 0.05))
    alpha = float(summary.get("alpha", 0.05))

    lines: List[str] = []
    lines.append("# Hybrid-KV (paper §7.7) decision-overhead measurement")
    lines.append("")
    lines.append("> **DISCLOSURE — read this BEFORE any number below.**")
    lines.append(">")
    for chunk in _wrap_paragraph(KV_REUSE_CAVEAT, width=80):
        lines.append(f"> {chunk}")
    lines.append("")
    lines.append(
        "This report is the *honest* counterpart to the README's "
        "Implementation Status row that flags Hybrid-KV as "
        "`⚠️ decision-plumbed; KV-reuse pending`."
    )
    lines.append("")
    lines.append(
        "What follows is what the local pipeline CAN measure today: the "
        "wall-clock cost of consulting `HybridKVManager` on every leaf "
        "(decision overhead) plus the cache statistics surfaced by "
        "`HybridKVManager.report()`. The cache-HIT fast path is documented "
        "as future work in `cts/eval/cuda_graph_skeleton.py` and the TODO "
        "block in `cts/mcts/hybrid_kv.py::HybridKVManager.__init__`."
    )
    lines.append("")
    lines.append("## 1. Configuration")
    lines.append("")
    lines.append(f"- seeds: {summary.get('n_seeds', 0)}")
    lines.append(f"- problems: {summary.get('n_problems', 0)}")
    lines.append(
        f"- TOST equivalence margin: ±{margin_frac * 100:.1f} % of "
        f"`hybrid_off` mean (α = {alpha:.3f})"
    )
    lines.append("")
    lines.append("## 2. Per-mode wall-clock (mean ± std)")
    lines.append("")
    lines.append(
        "| mode | n | wall_seconds (mean ± std) | decision_calls (mean) | "
        "cached_nodes (mean) | vram_used_gb (mean) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for mode in ALL_MODES:
        row = by_mode.get(mode, {})
        n = int(row.get("n", 0))
        m = float(row.get("wall_seconds_mean", 0.0))
        s = float(row.get("wall_seconds_std", 0.0))
        dc = float(row.get("decision_calls_mean", 0.0))
        cn = float(row.get("cached_nodes_mean", 0.0))
        vg = float(row.get("vram_used_gb_mean", 0.0))
        lines.append(
            f"| `{mode}` | {n} | {m:.4f} ± {s:.4f} | {dc:.1f} | "
            f"{cn:.1f} | {vg:.6f} |"
        )
    lines.append("")
    lines.append(
        "_Note: `cached_nodes` and `vram_used_gb` are expected to be **0** "
        "today because the cache HIT path is not yet plumbed. Non-zero "
        "values would indicate the post-submission `past_key_values` "
        "serialization has landed._"
    )
    lines.append("")
    lines.append("## 3. TOST equivalence verdict (hybrid_off vs hybrid_decision_only)")
    lines.append("")
    lines.append(f"- delta (absolute):    {float(tost.get('delta', 0.0)):.6f} s")
    lines.append(f"- mean_diff (off − on): {float(tost.get('mean_diff', 0.0)):.6f} s")
    lines.append(f"- p_lower:             {float(tost.get('p_lower', 1.0)):.6f}")
    lines.append(f"- p_upper:             {float(tost.get('p_upper', 1.0)):.6f}")
    lines.append(f"- p_max:               {float(tost.get('p_max', 1.0)):.6f}")
    lines.append(
        f"- **equivalent at α = {alpha:.3f}: "
        f"{bool(tost.get('equivalent', False))}**"
    )
    lines.append("")
    lines.append("## 4. What this report DOES NOT claim")
    lines.append("")
    lines.append(
        "- The paper's **−21 % wall-clock figure (§7.7)** is the reference "
        "number, not a measured local result on this machine. Measuring it "
        "requires the cache-HIT path documented as future work in "
        "`cts/eval/cuda_graph_skeleton.py`."
    )
    lines.append(
        "- The TOST verdict above is a *decision-overhead equivalence* "
        "test, not an accuracy-equivalence test. Once the HIT path is "
        "plumbed, reviewers should re-run this scaffold against per-seed "
        "accuracy arrays to reproduce the §7.7 'accuracy unchanged "
        "(p=0.89)' claim."
    )
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. JSONL persistence (used by the CLI)
# ---------------------------------------------------------------------------

def write_trace_jsonl(df: pd.DataFrame, out_path: Union[Path, str]) -> Path:
    """Persist the long-form DataFrame as JSONL (one row per line).
    Returns the resolved path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


__all__ = [
    "KV_REUSE_CAVEAT",
    "MODE_OFF",
    "MODE_DECISION_ONLY",
    "ALL_MODES",
    "LONG_FORM_COLUMNS",
    "tost_equivalence",
    "measure_decision_overhead",
    "summarize_hybrid_kv",
    "render_hybrid_kv_markdown",
    "write_trace_jsonl",
]
