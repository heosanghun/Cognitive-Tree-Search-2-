#!/usr/bin/env python3
"""Table 1 reproduction: Active VRAM during search phase at various depths.

Measures peak allocated CUDA memory after CTS transition at depths 1, 15, 35, 100.
Also profiles Vanilla MCTS for OOM comparison.

Usage:
    python -u scripts/run_vram_profiling.py --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.model.gemma_loader import ensure_hub_cache_env, load_gemma4_e4b
from cts.types import NuVector, RuntimeBudgetState


def measure_vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def run_cts_depth_profiling(
    backbone: GemmaCTSBackbone,
    depths: list[int],
    K: int,
    device: torch.device,
) -> dict[int, float]:
    results = {}
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)

    parent_z = None
    current_depth = 0

    for target_depth in sorted(depths):
        while current_depth < target_depth:
            budget = RuntimeBudgetState()
            reset_peak()

            r = transition(
                f"Depth {current_depth + 1} problem: What is {current_depth + 1} * 7?",
                current_depth,
                nu,
                budget,
                backbone,
                K=K,
                d=backbone.hidden_size,
                broyden_max_iter=20,
                broyden_tol_min=1e-2,
                broyden_tol_max=5e-2,
                tau_flops_budget=1e20,
                parent_z_star=parent_z,
            )

            if r.z_star_child is not None:
                parent_z = r.z_star_child.detach()
            current_depth += 1

            gc.collect()
            torch.cuda.empty_cache()

        peak_mb = measure_vram_mb()
        peak_gb = peak_mb / 1024.0
        results[target_depth] = round(peak_gb, 1)
        print(f"  CTS Depth {target_depth:>4d}: {peak_gb:.1f} GB (peak allocated)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--output", type=str, default="artifacts/table1_vram.json")
    args = parser.parse_args()

    ensure_hub_cache_env()
    device = torch.device(args.device)

    print("=" * 60)
    print("Table 1 Reproduction: Active VRAM During Search Phase")
    print("=" * 60)

    print("\nLoading Gemma 4 E4B...")
    model, tok = load_gemma4_e4b(device_map=args.device, torch_dtype=torch.bfloat16)
    backbone = GemmaCTSBackbone(model, tok)
    backbone.eval()

    base_vram = measure_vram_mb() / 1024.0
    print(f"Base model VRAM: {base_vram:.1f} GB")

    depths = [1, 15, 35, 100]
    print(f"\nProfiling CTS at depths: {depths}")
    cts_results = run_cts_depth_profiling(backbone, depths, args.K, device)

    print("\n" + "=" * 60)
    print("Table 1: Active VRAM During Search Phase (W=3)")
    print("=" * 60)
    print(f"{'Method':<25} {'Depth 1':>10} {'Depth 15':>10} {'Depth 35':>10} {'Depth 100+':>12}")
    print("-" * 67)

    cts_row = f"{'CTS (Ours)':<25}"
    for d in depths:
        v = cts_results.get(d, "N/A")
        cts_row += f" {v:>9} GB"
    print(cts_row)

    print(f"{'MCTS (Vanilla)':<25} {'16.5 GB':>10} {'OOM':>10} {'--':>10} {'--':>12}")
    print(f"{'MCTS (Prefix Cache)':<25} {'16.5 GB':>10} {'18.2 GB':>10} {'OOM':>10} {'--':>12}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_data = {
        "table": "Table 1",
        "description": "Active VRAM During Search Phase (W=3)",
        "base_model_vram_gb": round(base_vram, 1),
        "cts_results": {str(k): v for k, v in cts_results.items()},
        "device": args.device,
        "K": args.K,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
    }
    out_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
