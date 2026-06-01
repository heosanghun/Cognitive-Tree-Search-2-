#!/usr/bin/env python3
"""
CTS single transition on local Gemma 4 weights (E4B-it recommended).

Staged execution (recommended):
  **Stage 1 -- blend** (lightweight; verifies pipeline / Broyden convergence)
      python scripts/run_cts_local_gemma.py
      # or explicitly:  --map blend

  **Stage 2 -- parallel** (closer to paper Eq.(5) sparse module map; higher
  GPU cost per Broyden iteration)
      python scripts/run_cts_local_gemma.py --parallel
      # or:  --map parallel

Environment variables:
  CTS_GEMMA_MODEL_DIR   default: <repo_root>/gemma-4-E4B-it
  CTS_DEQ_MAP_MODE      used only when CLI args are absent (--map / --parallel
                        take precedence)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.model.gemma_loader import load_gemma4_e4b
from cts.types import NuVector, RuntimeBudgetState


def main() -> None:
    parser = argparse.ArgumentParser(description="CTS local Gemma — staged blend → parallel")
    parser.add_argument(
        "--map",
        choices=("blend", "full", "parallel"),
        default=None,
        help="Inner DEQ map (default: blend = Stage 1)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Shortcut for Stage 2: same as --map parallel",
    )
    parser.add_argument(
        "--max-decode",
        type=int,
        default=16,
        help="AR decode tokens from z* after DEQ (Gemma only; 1=greedy single token)",
    )
    args = parser.parse_args()

    if args.parallel:
        mode = "parallel"
    elif args.map is not None:
        mode = args.map
    else:
        mode = os.environ.get("CTS_DEQ_MAP_MODE", "blend")

    os.environ["CTS_DEQ_MAP_MODE"] = mode

    default_model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "gemma-4-E4B-it")
    )
    local = os.environ.get("CTS_GEMMA_MODEL_DIR", default_model_dir)
    os.environ.setdefault("CTS_GEMMA_MODEL_DIR", local)
    dm = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("local_model=", local)
    print("device=", dm)
    print("CTS_DEQ_MAP_MODE=", mode, "(Stage 1=blend, Stage 2=parallel)")
    model, tok = load_gemma4_e4b(model_id=local, device_map=dm, torch_dtype=torch.bfloat16)
    bb = GemmaCTSBackbone(model, tok)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    r = transition(
        "2+2=?",
        0,
        nu,
        RuntimeBudgetState(),
        bb,
        K=4,
        d=bb.hidden_size,
        broyden_max_iter=15,
        broyden_tol_min=1e-1,
        broyden_tol_max=2e-1,
        tau_flops_budget=1e22,
        max_decode_tokens=args.max_decode,
    )
    print("prune", r.prune, "iters", r.solver_stats.get("iterations"), "child", r.child_text)


if __name__ == "__main__":
    main()
