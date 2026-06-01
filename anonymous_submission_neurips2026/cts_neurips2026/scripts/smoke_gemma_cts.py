#!/usr/bin/env python3
"""Smoke: load Gemma 4 E4B + one CTS transition (uses D:\\...\\.hf_cache if set)."""

from __future__ import annotations

import os
import sys

# Allow running without package install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.model.gemma_loader import ensure_hub_cache_env, load_gemma4_e4b
from cts.types import NuVector, RuntimeBudgetState


def main() -> None:
    # Stage 1: light blend inner map (avoid heavy parallel stack per Broyden iter)
    os.environ.setdefault("CTS_DEQ_MAP_MODE", "blend")
    ensure_hub_cache_env()
    print("HF_HUB_CACHE=", os.environ.get("HF_HUB_CACHE"))
    print("CTS_DEQ_MAP_MODE=", os.environ.get("CTS_DEQ_MAP_MODE"))
    # Full E4B is large; use GPU when available (same RNG device as z — see latent/bottleneck).
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model, tok = load_gemma4_e4b(device_map=device_map, torch_dtype=torch.bfloat16)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device_map != "cpu":
            torch.cuda.empty_cache()
            model, tok = load_gemma4_e4b(device_map="cpu", torch_dtype=torch.bfloat16)
        else:
            raise
    backbone = GemmaCTSBackbone(model, tok)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    budget = RuntimeBudgetState()
    r = transition(
        "What is 2+2? Answer briefly.",
        0,
        nu,
        budget,
        backbone,
        K=64,
        d=backbone.hidden_size,
        broyden_max_iter=40,
        broyden_tol_min=1e-2,
        broyden_tol_max=5e-2,
        tau_flops_budget=1e20,
    )
    print("prune", r.prune, "converged", r.solver_stats.get("converged"), "child", r.child_text)


if __name__ == "__main__":
    main()
