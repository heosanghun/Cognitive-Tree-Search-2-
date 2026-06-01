#!/usr/bin/env python3
"""Two-ply MCTS rollouts (mock). See `cts/mcts/deep_rollout.py`."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.mcts.deep_rollout import two_ply_mcts_rollouts
from cts.policy.meta_policy import MetaPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="solve: 3+4")
    ap.add_argument("--sims-root", type=int, default=3)
    ap.add_argument("--sims-child", type=int, default=3)
    ap.add_argument("-W", type=int, default=3)
    ap.add_argument("-d", type=int, default=32)
    ap.add_argument("--meta", action="store_true")
    args = ap.parse_args()

    mp = MetaPolicy(text_dim=64, W=args.W) if args.meta else None
    out = two_ply_mcts_rollouts(
        args.prompt,
        sims_root=args.sims_root,
        sims_child=args.sims_child,
        W=args.W,
        d=args.d,
        meta_policy=mp,
    )
    print("best_action", out.best_action)
    print("root_ns", out.root.ns, "root_Q", [round(q, 3) for q in out.root.qs])
    print("child_ns", out.child.ns, "child_Q", [round(q, 3) for q in out.child.qs])


if __name__ == "__main__":
    main()
