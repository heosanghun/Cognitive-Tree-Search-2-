"""E2E CTS transition on real Gemma 4 E4B when cache is available."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.slow


def _gemma_available() -> bool:
    try:
        from transformers import Gemma4ForConditionalGeneration  # noqa: F401
    except Exception:
        return False
    cache = os.environ.get("HF_HUB_CACHE", "")
    if not cache:
        return False
    # Rough check: model dir exists
    from pathlib import Path

    p = Path(cache) / "models--google--gemma-4-E4B"
    return p.is_dir()


@pytest.mark.skipif(not _gemma_available(), reason="Set HF_HUB_CACHE and download google/gemma-4-E4B")
def test_gemma_transition_cpu():
    from cts.backbone.gemma_adapter import GemmaCTSBackbone
    from cts.deq.transition import transition
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.types import NuVector, RuntimeBudgetState

    model, tok = load_gemma4_e4b(device_map="cpu", torch_dtype=torch.bfloat16)
    bb = GemmaCTSBackbone(model, tok)
    nu = NuVector(nu_tol=0.6, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    r = transition(
        "Say hello in one word.",
        0,
        nu,
        budget,
        bb,
        K=4,
        d=bb.hidden_size,
        broyden_max_iter=50,
        broyden_tol_min=5e-2,
        broyden_tol_max=1e-1,
        tau_flops_budget=1e22,
    )
    assert not r.prune
    assert r.child_text is not None
    assert r.solver_stats["converged"] is True
