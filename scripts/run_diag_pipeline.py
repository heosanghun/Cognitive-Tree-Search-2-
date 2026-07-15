#!/usr/bin/env python3
"""Automation pipeline for diag_harness_2x2.py.

Executes each cell as a fresh subprocess to ensure process isolation.
Performs reproducibility validation and prints unified effect reports.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

def run_cell(
    cell: str,
    benchmark: str,
    data_path: str,
    base_model: str,
    it_model: str,
    device: str,
    problem_ids_file: str | None,
    limit: int | None,
    out_dir: Path,
    extra_args: List[str]
) -> Path:
    out_file = out_dir / f"{cell}_{benchmark}.jsonl"
    cmd = [
        sys.executable,
        "scripts/diag_harness_2x2.py",
        "--cell", cell,
        "--benchmark", benchmark,
        "--data", data_path,
        "--device", device,
        "--base-model", base_model,
        "--it-model", it_model,
        "--out", str(out_file),
    ]
    if problem_ids_file:
        cmd += ["--problem-ids-file", problem_ids_file]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    cmd += extra_args

    print(f"\n>>> Running subprocess for cell: {cell} on {device}")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=False, check=True)
    return out_file

def load_results_for_reproducibility(path: Path) -> List[List[int]]:
    token_ids_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "_meta" in data:
                continue
            token_ids_list.append(data.get("raw_output_token_ids") or [])
    return token_ids_list

def load_accuracy(path: Path) -> float:
    n_ok = 0
    n_total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "_meta" in data:
                continue
            n_total += 1
            if data.get("graded_match") is True:
                n_ok += 1
    return n_ok / max(1, n_total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["math500", "gsm8k"])
    ap.add_argument("--data", required=True, help="Path to benchmark test data")
    ap.add_argument("--problem-ids-file", default=None, help="STRATIFIED problem IDs file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-model", default="google/gemma-4-E4B")
    ap.add_argument("--it-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--device", default="cuda:1", help="CTS dedicated GPU (0~3 only)")
    ap.add_argument("--out-dir", default="results/diag2x2")
    args, extra_args = ap.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = ["REF", "M0", "M1", "M2", "M3"]
    outputs: Dict[str, Path] = {}

    # 1. Run all cells in fresh processes
    for cell in cells:
        outputs[cell] = run_cell(
            cell=cell,
            benchmark=args.benchmark,
            data_path=args.data,
            base_model=args.base_model,
            it_model=args.it_model,
            device=args.device,
            problem_ids_file=args.problem_ids_file,
            limit=args.limit,
            out_dir=out_dir,
            extra_args=extra_args
        )

    # 2. Reproducibility check: Run M0 a second time
    print("\n--- Running Reproducibility Verification (M0 duplicate run) ---")
    m0_dup_file = out_dir / f"M0_repro_{args.benchmark}.jsonl"
    run_cell(
        cell="M0",
        benchmark=args.benchmark,
        data_path=args.data,
        base_model=args.base_model,
        it_model=args.it_model,
        device=args.device,
        problem_ids_file=args.problem_ids_file,
        limit=args.limit,
        out_dir=out_dir,
        extra_args=extra_args + ["--out", str(m0_dup_file)]
    )

    m0_run1 = load_results_for_reproducibility(outputs["M0"])
    m0_run2 = load_results_for_reproducibility(m0_dup_file)

    bit_identical = (m0_run1 == m0_run2)
    print(f"Reproducibility verification bit-identical check: {'SUCCESS (Bit-identical)' if bit_identical else 'FAILED (Mismatched outputs)'}")

    # 3. Calculate and report effects
    accs: Dict[str, float] = {}
    for cell, path in outputs.items():
        accs[cell] = load_accuracy(path)

    m0, m1, m2, m3 = accs["M0"], accs["M1"], accs["M2"], accs["M3"]
    ref = accs["REF"]

    print("\n" + "="*50)
    print("           DIAGNOSTIC HARNESS 2x2 REPORT")
    print("="*50)
    print(f"Benchmark: {args.benchmark}")
    print(f"REF (Current Path accuracy):   {ref * 100:.2f}%")
    print(f"M0 (Base, Plain format):       {m0 * 100:.2f}%")
    print(f"M1 (Base, Canonical format):   {m1 * 100:.2f}%")
    print(f"M2 (IT, Plain format):         {m2 * 100:.2f}%")
    print(f"M3 (IT, Canonical candidate):  {m3 * 100:.2f}%")
    print("-"*50)
    print("Attributed Effects Analysis:")
    print(f"  Format Effect on Base Weights (M1 - M0): {((m1 - m0) * 100):+.2f}%")
    print(f"  Format Effect on IT Weights   (M3 - M2): {((m3 - m2) * 100):+.2f}%")
    print(f"  Weights Effect on Plain Format (M2 - M0): {((m2 - m0) * 100):+.2f}%")
    print(f"  Weights Effect on Canonical    (M3 - M1): {((m3 - m1) * 100):+.2f}%")
    
    # Interaction = (M3 - M2) - (M1 - M0)
    interaction = (m3 - m2) - (m1 - m0)
    print(f"  Interaction Effect (Synergy):             {(interaction * 100):+.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
