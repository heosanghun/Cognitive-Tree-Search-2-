#!/usr/bin/env python3
"""
Run CTS transition ablations (configs merge with default.yaml).

  python scripts/run_ablations.py
  python scripts/run_ablations.py --config ablation_no_ach
  python scripts/run_ablations.py --config ablation_static_5ht
  python scripts/run_ablations.py --routing dense   # CLI override when no yaml routing_mode
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState
from cts.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--routing",
        choices=("sparse", "dense"),
        default="sparse",
        help="Used only if merged config has no routing_mode",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="YAML name without path (e.g. ablation_no_ach) — merged with configs/default.yaml",
    )
    ap.add_argument("--static-5ht", action="store_true", help="Set ν_5HT to 1.0 (ignored if yaml sets nu_expl_static)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}
    routing_mode = cfg.get("routing_mode", args.routing)
    if "nu_expl_static" in cfg:
        nu_expl = float(cfg["nu_expl_static"])
    else:
        nu_expl = 1.0 if args.static_5ht else 0.65

    bb = MockTinyBackbone(hidden=64, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=nu_expl)
    budget = RuntimeBudgetState()
    r = transition(
        "ablation prompt",
        0,
        nu,
        budget,
        bb,
        K=64,
        d=64,
        broyden_max_iter=40,
        routing_mode=routing_mode,
        tau_flops_budget=1e20,
    )
    print(
        "config=",
        args.config,
        "routing_mode=",
        routing_mode,
        "nu_expl=",
        nu_expl,
        "prune=",
        r.prune,
        "converged=",
        r.solver_stats.get("converged"),
    )


if __name__ == "__main__":
    main()
