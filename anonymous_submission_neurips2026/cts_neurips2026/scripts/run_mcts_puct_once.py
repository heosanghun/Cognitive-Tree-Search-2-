#!/usr/bin/env python3
"""
PUCT → pick one child → one `transition` (mock backbone). Optional MetaPolicy for ν + priors.

  python scripts/run_mcts_puct_once.py
  python scripts/run_mcts_puct_once.py --meta   # MetaPolicy forward → NuVector + priors
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.mcts.episode import puct_select_and_expand_once
from cts.policy.meta_policy import MetaPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="2+2=?")
    ap.add_argument("-W", type=int, default=3)
    ap.add_argument("-K", type=int, default=4)
    ap.add_argument("-d", type=int, default=32)
    ap.add_argument(
        "--meta",
        action="store_true",
        help="Use MetaPolicy to produce NuVector + branch priors (paper Sec 4.1 stub)",
    )
    args = ap.parse_args()

    mp = MetaPolicy(text_dim=64, hidden=32, W=args.W) if args.meta else None
    out = puct_select_and_expand_once(
        args.prompt,
        W=args.W,
        K=args.K,
        d=args.d,
        meta_policy=mp,
    )
    print("selected_action=", out.selected_action)
    print("nu=", out.nu)
    print("priors=", [round(p, 4) for p in out.priors])
    print("converged=", out.transition.solver_stats.get("converged"), "prune=", out.transition.prune)
    print("child_text=", (out.transition.child_text or "")[:120])


if __name__ == "__main__":
    main()
