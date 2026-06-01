from cts.eval.flops_contract import (
    SOLVER_STATS_KEYS_TRANSITION,
    iso_flop_report_keys,
    public_isoflop_report,
)
from cts.types import NuVector, RuntimeBudgetState


def test_transition_solver_stats_covers_flop_contract():
    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.deq.transition import transition

    bb = MockTinyBackbone(hidden=64, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    r = transition("x", 0, nu, RuntimeBudgetState(), bb, K=4, d=64, broyden_max_iter=5)
    keys = set(r.solver_stats.keys())
    assert SOLVER_STATS_KEYS_TRANSITION <= keys


def test_public_report_matches_format_isoflop():
    stats = {
        "iterations": 2,
        "flops_inner_once": 3.0,
        "flops_broyden_estimate": 12.0,
        "phi_evals_per_broyden_iter": 2,
        "converged": True,
        "residual_norm": 0.01,
    }
    r1 = public_isoflop_report(stats)
    assert set(r1.keys()) == iso_flop_report_keys()
