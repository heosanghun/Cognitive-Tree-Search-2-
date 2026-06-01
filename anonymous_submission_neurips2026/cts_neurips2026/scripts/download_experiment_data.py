#!/usr/bin/env python3
"""
Download protocol-aligned datasets into `data/` (local, under repo or D:).

- **MATH-500** (`HuggingFaceH4/MATH-500`): full test split (500 rows) -> JSONL.
- **OpenMathInstruct-2** (`nvidia/OpenMathInstruct-2`, paper §6.1): streaming subset -> JSONL
- **MATH train prompts** (`EleutherAI/hendrycks_math`): up to 5000 training problems for Stage2-style PPO prompts

Requires: pip install datasets

Usage:
  python scripts/download_experiment_data.py
  python scripts/download_experiment_data.py --openmath-rows 100000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def _repo_data_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_math500(out_dir: Path) -> Path:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = out_dir / "math500" / "test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            row = {
                "problem": ex["problem"],
                "answer": ex["answer"],
                "solution": ex.get("solution", ""),
                "subject": ex.get("subject", ""),
                "level": ex.get("level", ""),
                "unique_id": ex.get("unique_id", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def download_openmath_subset(out_dir: Path, n_rows: int) -> Path:
    # Paper §6.1 specifies OpenMathInstruct-2 as the Stage-1 SFT corpus.
    # The schema accessor `prompt_text_from_openmath_row()` is dual-compatible
    # (handles both `question` and `problem` keys), so a switch from v1 to v2
    # is a one-line change here.
    stream = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    out = out_dir / "openmath_instruct" / f"train_{n_rows}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in stream:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
            if count >= n_rows:
                break
    if count < n_rows:
        raise RuntimeError(f"Only {count} rows available before stream ended")
    return out


def download_math_train_prompts(out_dir: Path, n_rows: int = 5000) -> Path:
    """
    Stage2-style: `configs/default.yaml` `stage2_math_prompts_n: 5000`.
    Aggregates EleutherAI/hendrycks_math train splits across subjects.
    """
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    out = out_dir / "stage2" / f"math_train_prompts_{n_rows}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for sub in subjects:
            ds = load_dataset("EleutherAI/hendrycks_math", sub, split="train")
            for ex in ds:
                row = {
                    "prompt": ex["problem"],
                    "subject": sub,
                    "level": ex.get("level", ""),
                    "type": ex.get("type", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                if count >= n_rows:
                    return out
    if count < n_rows:
        raise RuntimeError(f"Only collected {count} < {n_rows} from hendrycks_math train")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openmath-rows", type=int, default=100_000, help="OpenMathInstruct-2 streaming cap (paper §6.1)")
    ap.add_argument("--math-train-rows", type=int, default=5000, help="hendrycks MATH train prompts for Stage2")
    ap.add_argument("--skip-math500", action="store_true")
    ap.add_argument("--skip-openmath", action="store_true")
    ap.add_argument("--skip-math-train", action="store_true")
    args = ap.parse_args()

    data_dir = _repo_data_dir()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if not args.skip_math500:
        p = download_math500(data_dir)
        print("Wrote", p, "(500 lines)")
    if not args.skip_openmath:
        p = download_openmath_subset(data_dir, args.openmath_rows)
        print("Wrote", p, f"({args.openmath_rows} lines, streamed)")
    if not args.skip_math_train:
        p = download_math_train_prompts(data_dir, args.math_train_rows)
        print("Wrote", p, f"({args.math_train_rows} lines)")
    print("Done. See data/README.md")


if __name__ == "__main__":
    main()
