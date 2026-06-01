#!/usr/bin/env python3
"""Stage 1: OpenMath JSONL + Gemma + fixed-point surrogate (see `cts/train/stage1_openmath_train.py`)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.train.stage1_openmath_train import run_stage1_openmath_training


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="OpenMath JSONL (default: configs/data_paths.yaml openmath_train_jsonl)",
    )
    p.add_argument("--config", type=str, default="default")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lora", action="store_true", default=True,
                   help="Apply PEFT LoRA r=8 on language_model (paper §6.1, default: on)")
    p.add_argument("--no-lora", dest="lora", action="store_false",
                   help="Disable LoRA (ablation only)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--resume", action="store_true", help="Resume from artifacts/stage1_last.pt")
    p.add_argument("--model-dir", type=str, default=None, help="Override CTS_GEMMA_MODEL_DIR")
    args = p.parse_args()

    data_path = args.data
    if data_path is None:
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1]
        dp = yaml.safe_load((root / "configs" / "data_paths.yaml").read_text(encoding="utf-8"))
        data_path = str(root / dp.get("openmath_train_jsonl", "data/openmath_instruct/train_100000.jsonl"))

    if args.model_dir:
        os.environ["CTS_GEMMA_MODEL_DIR"] = args.model_dir

    run_stage1_openmath_training(
        openmath_jsonl=data_path,
        config_name=args.config,
        max_steps=args.max_steps,
        device=args.device,
        lora=args.lora,
        log_every=args.log_every,
        model_dir=args.model_dir,
        resume=args.resume,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
