#!/usr/bin/env python3
"""CTS episode wall-clock benchmark (CPU MockTinyBackbone, no GPU needed).

Mirrors the `scripts/run_cts_eval_full.py` episode invocation (K=64, W=3,
broyden_max_iter=20, FAISS context, per-episode seeds) on the CPU mock
backbone, so the framework hot path (PUCT loop + DEQ dense-Broyden solves +
critic/meta-policy calls) can be profiled and regression-tested on any
machine. Used for the 2026-07 performance-optimization experiments
(`results/perf_opt/`): the controlled 2-simulation workload at K=64/d=64
went from 24.2 s (HEAD baseline) to 4.1 s after the Sherman-Morrison
inverse-Jacobian Broyden rewrite + context-encoding hoist.

Examples:
  # controlled workload used in results/perf_opt/BENCHMARK.md
  python scripts/bench_episode_perf.py --episodes 1 --K 64 --d 64 --sims 2

  # eval-protocol episode (tau=1e13, 180 s wall cap, tau-driven sims)
  python scripts/bench_episode_perf.py --episodes 1 --K 64 --d 64 \
      --tau 1e13 --wall 180
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.cts_episode import cts_full_episode
from cts.policy.meta_policy import MetaPolicy


class DecodingMock(MockTinyBackbone):
    def decode_from_z_star(self, z_star, *, max_new_tokens=64, problem_text=None):
        head = z_star.detach().float().mean().item()
        return f"answer={head:+.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--W", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--tau", type=float, default=2e13)
    ap.add_argument("--sims", type=int, default=None, help="w_override: cap outer PUCT loop")
    ap.add_argument("--wall", type=float, default=None, help="wall_clock_budget_s per episode")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(2026)
    bb = DecodingMock(hidden=args.d, num_layers=8)
    meta = MetaPolicy(text_dim=args.d, hidden=64, W=args.W)
    critic = NeuroCritic(z_dim=args.d)

    prompts = [
        "Find the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square.",
        "Let x, y, z be positive reals with xyz = 1. Minimize x + y + z subject to x^2+y^2+z^2=9.",
        "A fair coin is flipped 10 times. What is the probability of at least 7 heads?",
    ]

    rows = []
    total = 0.0
    for ep in range(args.episodes):
        q = prompts[ep % len(prompts)]
        faiss_ctx = LatentContextWindow(dim=args.d, retrieval_k=3, min_steps=10)
        t0 = time.perf_counter()
        result = cts_full_episode(
            q,
            backbone=bb,
            meta_policy=meta,
            critic=critic,
            W=args.W,
            K=args.K,
            tau_budget=args.tau,
            broyden_max_iter=args.max_iter,
            broyden_tol_min=1e-4,
            broyden_tol_max=1e-2,
            top_k=3,
            puct_variant="paper",
            faiss_context=faiss_ctx,
            max_decode_tokens=64,
            device=torch.device("cpu"),
            wall_clock_budget_s=args.wall,
            z0_seed=100_000 + ep,
            selection_seed=100_001 + ep,
            nu_config_mode="4nu",
            w_override=args.sims,
        )
        dt = time.perf_counter() - t0
        total += dt
        z = result.best_z_star
        zh = hashlib.sha256(z.detach().float().numpy().tobytes()).hexdigest()[:16] if z is not None else "none"
        row = {
            "episode": ep,
            "wall_s": round(dt, 3),
            "answer": result.answer,
            "tree_size": result.stats["tree_size"],
            "max_depth": result.stats["max_depth"],
            "sim_count": result.stats["sim_count"],
            "total_mac": result.total_mac,
            "total_iterations": result.total_iterations,
            "z_star_sha": zh,
        }
        rows.append(row)
        print(f"[ep {ep}] wall={dt:.2f}s tree={row['tree_size']} depth={row['max_depth']} "
              f"sims={row['sim_count']} iters={row['total_iterations']} ans={result.answer} z={zh}", flush=True)

    mean = total / max(1, args.episodes)
    print(f"\nMEAN episode wall-clock: {mean:.2f}s over {args.episodes} episodes "
          f"(K={args.K}, d={args.d}, W={args.W}, max_iter={args.max_iter})")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"mean_wall_s": mean, "config": vars(args), "episodes": rows}, f, indent=2)


if __name__ == "__main__":
    main()
