import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState


def test_transition_mock_converges():
    bb = MockTinyBackbone(hidden=32, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    r = transition(
        "hello",
        0,
        nu,
        budget,
        bb,
        K=4,
        d=32,
        broyden_max_iter=40,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        tau_flops_budget=1e20,
    )
    assert not r.prune
    assert r.child_text is not None
    assert r.solver_stats["converged"] is True
    assert r.z_star_child.shape == (4, 32)
