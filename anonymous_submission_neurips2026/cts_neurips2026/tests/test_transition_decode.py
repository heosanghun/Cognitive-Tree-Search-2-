"""transition() passes max_decode_tokens to backbones that support it."""

from __future__ import annotations

import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState


class _BackboneWithDecode(MockTinyBackbone):
    def __init__(self) -> None:
        super().__init__(hidden=32, num_layers=8)
        self.last_max_new_tokens: int | None = None

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 1) -> str:
        self.last_max_new_tokens = max_new_tokens
        return "ok"


def test_transition_passes_max_decode_tokens():
    bb = _BackboneWithDecode()
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    r = transition(
        "x",
        0,
        nu,
        budget,
        bb,
        K=4,
        d=32,
        broyden_max_iter=30,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        tau_flops_budget=1e20,
        max_decode_tokens=17,
    )
    assert not r.prune
    assert bb.last_max_new_tokens == 17
    assert r.child_text == "ok"
