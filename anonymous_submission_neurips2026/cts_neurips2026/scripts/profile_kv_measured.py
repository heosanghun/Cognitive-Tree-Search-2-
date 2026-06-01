#!/usr/bin/env python3
"""
GPU **measured** KV tensor peak (bf16 K/V per layer) vs tree depth proxy.

Requires CUDA. Writes CSV like `profile_vram_latency` for merging with analytic rows.

  python scripts/profile_kv_measured.py --depths 1 3 5 --out artifacts/kv_measured.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

from cts.baselines.mcts_kv_baseline import KVRetentionConfig
from cts.eval.kv_measured import sweep_kv_measured_rows
from cts.perf.profiler import write_sweep_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--out", type=Path, default=Path("artifacts/kv_measured.csv"))
    ap.add_argument("--tokens-per-depth", type=int, default=None)
    args = ap.parse_args()

    cfg = KVRetentionConfig()
    if args.tokens_per_depth is not None:
        cfg = KVRetentionConfig(tokens_per_depth_step=args.tokens_per_depth)
    rows = sweep_kv_measured_rows(args.depths, cfg)
    write_sweep_csv(rows, args.out)
    print(f"Wrote {args.out}")
    if rows and rows[0].get("peak_vram_gb") is None:
        print("Note: CUDA not available - CSV has null peaks (run on GPU for measured VRAM).")


if __name__ == "__main__":
    main()
