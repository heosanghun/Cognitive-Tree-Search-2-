"""Tests for parallel batch DEQ transition (paper §4.1)."""

import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition_batch
from cts.types import NuVector, RuntimeBudgetState


def test_transition_batch_returns_w_results():
    bb = MockTinyBackbone(hidden=32, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    results = transition_batch(
        "test prompt", nu, budget, bb,
        W=3, K=4, d=32, broyden_max_iter=10,
    )
    assert len(results) == 3


def test_transition_batch_branches_differ():
    bb = MockTinyBackbone(hidden=32, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    results = transition_batch(
        "test prompt", nu, budget, bb,
        W=3, K=4, d=32, broyden_max_iter=10,
    )
    texts = [r.child_text for r in results if r.child_text]
    if len(texts) > 1:
        assert len(set(texts)) > 1, "Branches should produce different texts"


def test_transition_batch_all_converge():
    bb = MockTinyBackbone(hidden=32, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    results = transition_batch(
        "test prompt", nu, budget, bb,
        W=3, K=4, d=32, broyden_max_iter=40,
    )
    for r in results:
        assert r.solver_stats["converged"]
