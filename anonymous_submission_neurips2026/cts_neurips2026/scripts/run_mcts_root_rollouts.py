#!/usr/bin/env python3
"""Several root PUCT simulations with mean-Q backup (mock backbone)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cts.critic.neuro_critic import NeuroCritic
from cts.mcts.critic_reward import make_critic_reward_fn
from cts.mcts.episode import mcts_root_rollouts
from cts.policy.meta_policy import MetaPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="1+1=?")
    ap.add_argument("--sims", type=int, default=6)
    ap.add_argument("-W", type=int, default=3)
    ap.add_argument("-K", type=int, default=4)
    ap.add_argument("-d", type=int, default=32)
    ap.add_argument("--meta", action="store_true")
    ap.add_argument(
        "--critic",
        action="store_true",
        help="Reward = sigmoid(NeuroCritic(z*)) instead of DEQ convergence",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    mp = MetaPolicy(text_dim=64, W=args.W) if args.meta else None
    reward_fn = None
    if args.critic:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cr = NeuroCritic(args.d).to(dev)
        reward_fn = make_critic_reward_fn(cr, z_dim=args.d, device=dev)

    out = mcts_root_rollouts(
        args.prompt,
        num_simulations=args.sims,
        W=args.W,
        K=args.K,
        d=args.d,
        meta_policy=mp,
        reward_fn=reward_fn,
    )
    payload = {
        "ns": out.ns,
        "qs": [round(q, 4) for q in out.qs],
        "priors": [round(p, 4) for p in out.priors],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("ns", out.ns, "mean_Q", [round(q, 4) for q in out.qs])


if __name__ == "__main__":
    main()
