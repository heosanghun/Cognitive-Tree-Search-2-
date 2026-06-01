"""
Iso-FLOP budget matching (paper Sec. 7.3): align compute across CTS and baselines.

Uses the same MAC table as `transition()` (`cts/routing/lut_mac.json`) and optional
Broyden iteration count to approximate FLOPs per transition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _lut_path() -> Path:
    return Path(__file__).resolve().parents[1] / "routing" / "lut_mac.json"


def load_mac_per_module() -> List[float]:
    with _lut_path().open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["mac_per_module"])


def estimate_sparse_step_flops(
    module_weights: Sequence[float],
    macs_per_module: Sequence[float] | None = None,
    *,
    nu_act: float = 1.0,
) -> float:
    """FLOPs for one DEQ inner step after routing (same structure as `transition`)."""
    macs = macs_per_module or load_mac_per_module()
    if len(macs) != len(module_weights):
        raise ValueError("macs_per_module length must match module_weights")
    total = 0.0
    for i, w in enumerate(module_weights):
        total += float(w) * float(macs[i]) * nu_act
    return total


def estimate_transition_flops_from_stats(
    solver_stats: Dict[str, Any],
    macs_per_module: Sequence[float] | None = None,
) -> float:
    """Use recorded `flops_used` when present; else 0."""
    v = solver_stats.get("flops_used")
    if v is not None:
        return float(v)
    return 0.0


def estimate_query_flops_stub() -> float:
    """Deprecated name — prefer `estimate_sparse_step_flops` or stats-based totals."""
    return 0.0


def estimate_broyden_flops_from_inner(
    flops_inner_once: float,
    broyden_iterations: int,
    *,
    phi_evals_per_broyden_iter: int = 2,
) -> float:
    """Approximate total φ-eval cost during Broyden (matches `transition` solver_stats)."""
    return float(flops_inner_once) * float(max(1, broyden_iterations)) * float(phi_evals_per_broyden_iter)


def format_isoflop_report(solver_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Stable keys for CLI / JSON (Iso-FLOP protocol)."""
    inner = solver_stats.get("flops_inner_once", solver_stats.get("flops_used"))
    it = int(solver_stats.get("iterations") or 0)
    pe = int(solver_stats.get("phi_evals_per_broyden_iter") or 2)
    broyden_est = solver_stats.get("flops_broyden_estimate")
    if broyden_est is None and inner is not None:
        broyden_est = estimate_broyden_flops_from_inner(float(inner), it, phi_evals_per_broyden_iter=pe)
    return {
        "broyden_iterations": it,
        "phi_evals_per_broyden_iter": pe,
        "flops_inner_once": inner,
        "flops_broyden_estimate": broyden_est,
        "converged": solver_stats.get("converged"),
        "residual_norm": solver_stats.get("residual_norm"),
    }
