#!/usr/bin/env python3
"""Stage 2: MATH prompts JSONL + Gemma backbone + MetaPolicy PPO (see `cts/train/stage2_ppo_train.py`)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.train.stage2_ppo_train import run_stage2_math_ppo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="MATH train prompts JSONL (default: configs/data_paths.yaml stage2_math_prompts_jsonl)",
    )
    p.add_argument("--config", type=str, default="default")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--W", type=int, default=3)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--collect-batch", type=int, default=4)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--broyden-max-iter", type=int, default=12)
    p.add_argument("--parallel-map", action="store_true", help="CTS_DEQ_MAP_MODE=parallel (heavy)")
    p.add_argument("--stage1-ckpt", type=str, default=None, help="Optional artifacts/stage1_last.pt")
    p.add_argument("--use-critic-reward", action="store_true")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Override stage2_ppo_train save_every (default 1000). Use a smaller "
        "value (e.g. 20) for shorter compute-constrained runs so an intermediate "
        "checkpoint is recoverable if the long run is interrupted.",
    )
    p.add_argument("--model-dir", type=str, default=None)
    args = p.parse_args()

    data_path = args.data
    if data_path is None:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1]
        dp = yaml.safe_load((root / "configs" / "data_paths.yaml").read_text(encoding="utf-8"))
        data_path = str(root / dp.get("stage2_math_prompts_jsonl", "data/stage2/math_train_prompts_5000.jsonl"))

    if args.model_dir:
        os.environ["CTS_GEMMA_MODEL_DIR"] = args.model_dir

    kwargs: dict = {
        "math_prompts_jsonl": data_path,
        "config_name": args.config,
        "total_steps": args.steps,
        "device": args.device,
        "W": args.W,
        "K": args.K,
        "collect_batch": args.collect_batch,
        "ppo_epochs": args.ppo_epochs,
        "broyden_max_iter": args.broyden_max_iter,
        "parallel_map": args.parallel_map,
        "stage1_checkpoint": args.stage1_ckpt,
        "use_critic_reward": args.use_critic_reward,
        "log_every": args.log_every,
    }
    if args.save_every is not None:
        kwargs["save_every"] = int(args.save_every)
    run_stage2_math_ppo(**kwargs)


if __name__ == "__main__":
    main()
