#!/usr/bin/env python3
"""
Table-2 style full-cap MATH (500) + optional ARC JSONL (200) with structured JSON output.

Does not run by default in CI (long GPU time). Use after `download_experiment_data.py`.

  python scripts/run_table2_full_bench.py
  python scripts/run_table2_full_bench.py --arc-data path/to/arc.jsonl
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-data", type=str, default=str(ROOT / "data" / "math500" / "test.jsonl"))
    ap.add_argument("--math-limit", type=int, default=500)
    ap.add_argument("--arc-data", type=str, default=None)
    ap.add_argument("--arc-limit", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "artifacts"))
    ap.add_argument("--skip-gemma", action="store_true", help="Dry-run: print commands only")
    args = ap.parse_args()

    py = sys.executable
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CTS_GEMMA_MODEL_DIR", str(ROOT / "gemma-4-E4B-it"))

    math_out = out / "table2_math500_full.json"
    cmd_m = [
        py,
        str(ROOT / "scripts" / "run_math500.py"),
        "--data",
        args.math_data,
        "--gemma",
        "--limit",
        str(args.math_limit),
        "--think-prompt",
        "--chat-template",
        "--out-json",
        str(math_out),
    ]
    print("MATH:", " ".join(cmd_m))
    if not args.skip_gemma:
        subprocess.check_call(cmd_m, cwd=str(ROOT))

    if args.arc_data and Path(args.arc_data).is_file():
        arc_out = out / "table2_arc_full.json"
        cmd_a = [
            py,
            str(ROOT / "scripts" / "run_arc_agi_text.py"),
            "--data",
            args.arc_data,
            "--gemma",
            "--limit",
            str(args.arc_limit),
            "--think-prompt",
            "--chat-template",
            "--out-json",
            str(arc_out),
        ]
        print("ARC:", " ".join(cmd_a))
        if not args.skip_gemma:
            subprocess.check_call(cmd_a, cwd=str(ROOT))
    else:
        print("(no --arc-data or missing file; skip ARC)")

    manifest = out / "table2_full_bench_manifest.txt"
    manifest.write_text(
        f"math_limit={args.math_limit}\nmath_out={math_out}\narc={args.arc_data or 'skipped'}\n",
        encoding="utf-8",
    )
    print("Wrote", manifest)


if __name__ == "__main__":
    main()
