"""
Iso-FLOP accounting — **canonical field names** (single reference for tools & docs).

Flow:
1. `cts.deq.transition.transition()` fills `TransitionResult.solver_stats` with
   `SOLVER_STATS_KEYS_TRANSITION` (raw Broyden + LUT MAC totals).
2. `cts.eval.isoflop_matcher.format_isoflop_report(solver_stats)` returns the
   **public** JSON shape `ISO_FLOP_REPORT_KEYS` for CLI / `table2_isoflop_*.json`.

Always use `format_isoflop_report` for external reports; do not duplicate formulas elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from cts.eval.isoflop_matcher import format_isoflop_report

# Populated by `transition()` — keep in sync with `cts/deq/transition.py`
SOLVER_STATS_KEYS_TRANSITION: FrozenSet[str] = frozenset(
    {
        "iterations",
        "residual_norm",
        "converged",
        "flops_used",
        "flops_inner_once",
        "flops_broyden_estimate",
        "phi_evals_per_broyden_iter",
    }
)

# Optional keys after successful solve
SOLVER_STATS_OPTIONAL_KEYS: FrozenSet[str] = frozenset({"act_halt"})


def iso_flop_report_keys() -> FrozenSet[str]:
    """Keys returned by `format_isoflop_report` (stable protocol)."""
    sample = format_isoflop_report(
        {
            "iterations": 1,
            "flops_inner_once": 1.0,
            "flops_broyden_estimate": 2.0,
            "phi_evals_per_broyden_iter": 2,
            "converged": True,
            "residual_norm": 0.0,
        }
    )
    return frozenset(sample.keys())


def public_isoflop_report(solver_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Preferred alias: public Iso-FLOP JSON from raw `solver_stats`."""
    return format_isoflop_report(solver_stats)
