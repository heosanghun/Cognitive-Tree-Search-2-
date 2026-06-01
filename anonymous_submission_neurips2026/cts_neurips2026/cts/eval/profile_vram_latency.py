"""Depth sweep: peak VRAM and timing for CTS vs analytic KV-MCTS baseline (Table 1 style)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cts.baselines.mcts_kv_baseline import KVRetentionConfig, estimate_mcts_kv_peak_gb
from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.perf.profiler import run_timed, write_sweep_csv
from cts.types import NuVector, RuntimeBudgetState


def main() -> None:
    p = argparse.ArgumentParser(description="CTS vs KV-baseline profile sweep (Table 1 style)")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 5, 10, 15])
    p.add_argument("--out", type=Path, default=Path("artifacts/profile_table1.csv"))
    p.add_argument("--cuda", action="store_true")
    p.add_argument(
        "--skip-cts",
        action="store_true",
        help="Only emit analytic KV rows (no mock backbone timing)",
    )
    p.add_argument(
        "--kv-tokens-per-depth",
        type=int,
        default=None,
        help="Override KVRetentionConfig.tokens_per_depth_step",
    )
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    rows = []

    kv_cfg = KVRetentionConfig()
    if args.kv_tokens_per_depth is not None:
        kv_cfg = KVRetentionConfig(tokens_per_depth_step=args.kv_tokens_per_depth)

    for d in args.depths:
        kv_gb = estimate_mcts_kv_peak_gb(d, kv_cfg)
        rows.append(
            {
                "tree_depth_proxy": d,
                "approach": "mcts_kv_analytic",
                "peak_vram_gb": round(kv_gb, 4),
                "total_ms_3_branches": "",
                "latency_ms_per_node": "",
                "notes": "linear KV growth vs depth (analytic)",
            }
        )

    if args.skip_cts:
        write_sweep_csv(rows, args.out)
        print(f"Wrote {args.out}")
        return

    backbone = MockTinyBackbone(hidden=64, num_layers=42).to(device)
    nu = NuVector()

    for d in args.depths:
        text = "prompt " * d
        budget = RuntimeBudgetState()

        def one() -> None:
            nonlocal budget
            budget = RuntimeBudgetState()
            for b in range(3):
                r = transition(
                    text,
                    b,
                    nu,
                    budget,
                    backbone,
                    K=64,
                    d=64,
                    tau_flops_budget=1e18,
                )
                budget = r.budget

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        _, ms = run_timed(one, cuda_sync=True)
        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        per_node = ms / 3.0 if ms else 0.0
        rows.append(
            {
                "tree_depth_proxy": d,
                "approach": "cts_mock",
                "peak_vram_gb": round(peak_gb, 4),
                "total_ms_3_branches": round(ms, 3),
                "latency_ms_per_node": round(per_node, 3),
                "notes": "MockTinyBackbone DEQ transition ×3 branches",
            }
        )

    write_sweep_csv(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
