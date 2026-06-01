#!/usr/bin/env python3
"""
Orchestrate data download (optional) -> Stage 1 (OpenMath) -> Stage 2 (MATH PPO).

Does not run heavy jobs by default: use `--run` to execute; otherwise prints the recommended commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Actually run subprocesses (long-running on GPU)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--stage2-only", action="store_true")
    ap.add_argument("--openmath-rows", type=int, default=100_000)
    ap.add_argument("--stage1-extra", nargs="*", default=[], help="Extra args for run_stage1_openmath.py")
    ap.add_argument("--stage2-extra", nargs="*", default=[], help="Extra args for run_stage2_math_ppo.py")
    args = ap.parse_args()

    dl = [
        py,
        str(root / "scripts" / "download_experiment_data.py"),
        "--openmath-rows",
        str(args.openmath_rows),
    ]
    s1 = [py, str(root / "scripts" / "run_stage1_openmath.py"), *args.stage1_extra]
    s2 = [py, str(root / "scripts" / "run_stage2_math_ppo.py"), "--stage1-ckpt", str(root / "artifacts" / "stage1_last.pt"), *args.stage2_extra]

    if not args.run:
        print("Set HF_TOKEN if needed. Recommended sequence:\n")
        if not args.skip_download and not args.stage2_only:
            print(" ", " ".join(dl))
        if not args.stage2_only:
            print(" ", " ".join(s1))
        if not args.stage1_only:
            print(" ", " ".join(s2))
        print("\nRe-run with --run to execute (GPU time and disk).")
        return

    if not args.skip_download and not args.stage2_only:
        subprocess.check_call(dl, cwd=str(root))
    if not args.stage2_only:
        subprocess.check_call(s1, cwd=str(root))
    if not args.stage1_only:
        subprocess.check_call(s2, cwd=str(root))


if __name__ == "__main__":
    main()
