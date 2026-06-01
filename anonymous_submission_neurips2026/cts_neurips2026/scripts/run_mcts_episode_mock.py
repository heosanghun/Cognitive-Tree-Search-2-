#!/usr/bin/env python3
"""One root expansion: W parallel CTS transitions on MockTinyBackbone (MCTS wiring smoke)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.mcts.episode import expand_root_parallel_branches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="1+1=?")
    ap.add_argument("-W", type=int, default=3, dest="branching")
    ap.add_argument("-K", type=int, default=4, dest="soft_k")
    ap.add_argument("-d", type=int, default=32, dest="hidden")
    args = ap.parse_args()

    tree, results = expand_root_parallel_branches(
        args.prompt,
        W=args.branching,
        K=args.soft_k,
        d=args.hidden,
    )
    print("nodes=", len(tree.nodes), "root_children=", len(tree.root().children_ids))
    for i, r in enumerate(results):
        print(i, "prune", r.prune, "converged", r.solver_stats.get("converged"), "child", (r.child_text or "")[:80])


if __name__ == "__main__":
    main()
