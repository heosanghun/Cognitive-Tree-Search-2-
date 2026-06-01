#!/usr/bin/env python3
"""
Exit 0 if machine/repo is ready for paper-style runs; else print gaps.

Checks: optional GPU, disk headroom, CTS_GEMMA_MODEL_DIR or HF, key data paths from data_paths.yaml.
Does NOT validate HF_TOKEN (cannot know gated state without Hub call).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1])


def _need_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024**3)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-free-gb", type=float, default=18.0, help="Min free space on repo drive")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--require-gemma-dir", action="store_true", help="Require CTS_GEMMA_MODEL_DIR to exist")
    args = ap.parse_args()

    failed: list[str] = []
    warns: list[str] = []

    try:
        import torch

        if args.require_gpu and not torch.cuda.is_available():
            failed.append("CUDA not available (--require-gpu)")
        elif torch.cuda.is_available():
            print("OK: GPU", torch.cuda.get_device_name(0))
    except ImportError:
        failed.append("torch not installed")

    # Disk (Windows: drive of ROOT)
    try:
        import shutil

        usage = shutil.disk_usage(str(ROOT))
        free_gb = usage.free / (1024**3)
        if free_gb < args.min_free_gb:
            failed.append(f"free disk {free_gb:.1f} GB < {args.min_free_gb} GB")
        else:
            print(f"OK: free disk ~{free_gb:.1f} GB on repo volume")
    except OSError as e:
        warns.append(f"disk check: {e}")

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "")
    if mid:
        p = Path(mid)
        if p.is_dir() and (p / "config.json").is_file():
            print("OK: CTS_GEMMA_MODEL_DIR", mid, f"~{_need_gb(p):.1f} GB")
        elif args.require_gemma_dir:
            failed.append(f"CTS_GEMMA_MODEL_DIR invalid: {mid}")
        else:
            warns.append(f"CTS_GEMMA_MODEL_DIR set but folder/config missing: {mid}")
    else:
        warns.append("CTS_GEMMA_MODEL_DIR unset — will use Hub id (needs HF_TOKEN if gated)")

    if not os.environ.get("HF_TOKEN"):
        warns.append("HF_TOKEN unset — required if Gemma is gated on Hub")

    try:
        import yaml

        dp = yaml.safe_load((ROOT / "configs" / "data_paths.yaml").read_text(encoding="utf-8"))
        for key in ("math500_jsonl", "openmath_train_jsonl", "stage2_math_prompts_jsonl"):
            rel = dp.get(key)
            if not rel:
                continue
            path = ROOT / rel
            if path.is_file():
                print("OK: data", key, path)
            else:
                warns.append(f"missing data file {key}: {path} — run download_experiment_data.py")
    except Exception as e:
        warns.append(f"data_paths check: {e}")

    for w in warns:
        print("WARN:", w)
    for f in failed:
        print("FAIL:", f)

    if failed:
        return 1
    print("verify_repro_prereqs: all required checks passed (warnings may remain).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
