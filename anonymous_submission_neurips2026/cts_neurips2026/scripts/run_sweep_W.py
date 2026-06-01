#!/usr/bin/env python3
"""Paper Table 15: W = MCTS per-step simulation budget scaling sweep.

W here is the total number of PUCT-then-expand iterations executed per
``cts_full_episode`` call, NOT the per-leaf branching factor W=3 from
paper §4.1 (the latter is unaffected by this sweep — see
``cts/mcts/cts_episode.py`` docstring on ``w_override``).

Defaults reproduce paper Table 15:
  - W values: {4, 8, 16, 32, 64, 128}
  - benchmark: AIME (data/aime/test.jsonl)
  - method:    cts_4nu
  - seeds:     {0, 1, 2}
  - limit:     30 (full AIME 2026 test set)

CPU-only invocations:
    python scripts/run_sweep_W.py --dry-run --w-values 4 8 --seeds 0

Outputs:
  - results/sweep_W/sweep_W.jsonl
  - results/sweep_W/sweep_W.md
  - results/sweep_W/sweep_W_plan.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cts.eval.sweep_utils import (  # noqa: E402
    append_sweep_row,
    dry_run_grid,
    load_sweep_jsonl,
    render_sweep_markdown,
    summarize_sweep,
)


DEFAULT_W_VALUES: Tuple[int, ...] = (4, 8, 16, 32, 64, 128)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_BENCHMARK: str = "aime"
DEFAULT_METHOD: str = "cts_4nu"
DEFAULT_LIMIT: int = 30
OUT_DIR = ROOT / "results" / "sweep_W"
JSONL_PATH = OUT_DIR / "sweep_W.jsonl"
MD_PATH = OUT_DIR / "sweep_W.md"
PLAN_PATH = OUT_DIR / "sweep_W_plan.txt"


def _completed_pairs(jsonl_path: Path, param_name: str) -> set:
    rows = load_sweep_jsonl(jsonl_path)
    done = set()
    for r in rows:
        if param_name in r and "seed" in r:
            try:
                done.add((int(r[param_name]), int(r["seed"])))
            except (TypeError, ValueError):
                continue
    return done


def _patch_cts_full_episode_with_w_override(w_value: int):
    from cts.mcts import cts_episode as _ep
    real = _ep.cts_full_episode

    def _wrapped(*args, **kwargs):
        kwargs.setdefault("w_override", w_value)
        return real(*args, **kwargs)

    _ep.cts_full_episode = _wrapped  # type: ignore[assignment]
    return real


def _restore_cts_full_episode(real_fn) -> None:
    from cts.mcts import cts_episode as _ep
    _ep.cts_full_episode = real_fn  # type: ignore[assignment]


def plan_sweep(
    w_values: Sequence[int],
    seeds: Sequence[int],
    benchmark: str,
) -> List[Tuple[int, int, str]]:
    return dry_run_grid(w_values, seeds, [benchmark])  # type: ignore[return-value]


def write_plan(grid: Sequence[Tuple[int, int, str]], plan_path: Path) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        f.write(f"# W-sweep plan ({len(grid)} jobs)\n")
        f.write("# format: W seed benchmark\n")
        for w, s, b in grid:
            f.write(f"{w}\t{s}\t{b}\n")


def run_sweep(
    *,
    w_values: Sequence[int] = DEFAULT_W_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    benchmark: str = DEFAULT_BENCHMARK,
    method: str = DEFAULT_METHOD,
    limit: int = DEFAULT_LIMIT,
    config_name: str = "default",
    device: str = "cuda:0",
    model_dir: Optional[str] = None,
    jsonl_path: Path = JSONL_PATH,
    md_path: Path = MD_PATH,
    plan_path: Path = PLAN_PATH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    grid = plan_sweep(w_values, seeds, benchmark)
    write_plan(grid, plan_path)

    if dry_run:
        print(f"[dry-run] planned {len(grid)} (W, seed) eval jobs:")
        for w, s, b in grid:
            print(f"  W={w} seed={s} bench={b} method={method} limit={limit}")
        print(f"[dry-run] plan written to {plan_path}")
        print(f"[dry-run] no JSONL written; exiting before eval launch.")
        return {"dry_run": True, "planned_jobs": len(grid)}

    done = _completed_pairs(jsonl_path, "W")

    from scripts.run_cts_eval_full import run_single_evaluation  # noqa: E402

    launched = 0
    skipped = 0
    for w_val, seed, _bench in grid:
        if (w_val, seed) in done:
            skipped += 1
            continue
        print(f"[W-sweep] launching W={w_val} seed={seed} bench={benchmark} method={method}")
        real = _patch_cts_full_episode_with_w_override(w_val)
        try:
            t0 = time.time()
            res = run_single_evaluation(
                method=method,
                benchmark=benchmark,
                seed=int(seed),
                config_name=config_name,
                device=device,
                model_dir=model_dir,
                limit=int(limit),
            )
            elapsed = time.time() - t0
        finally:
            _restore_cts_full_episode(real)
        scores = res.get("scores") or []
        for pi, score in enumerate(scores):
            append_sweep_row(jsonl_path, {
                "W": int(w_val),
                "seed": int(seed),
                "benchmark": benchmark,
                "method": method,
                "problem_idx": pi,
                "score": float(score),
                "elapsed_s": float(elapsed),
            })
        if not scores:
            append_sweep_row(jsonl_path, {
                "W": int(w_val),
                "seed": int(seed),
                "benchmark": benchmark,
                "method": method,
                "problem_idx": -1,
                "score": float(res.get("accuracy", 0.0)),
                "elapsed_s": float(elapsed),
                "error": res.get("error"),
            })
        launched += 1

    rows = load_sweep_jsonl(jsonl_path)
    summary = summarize_sweep(rows, "W")
    render_sweep_markdown(
        summary, "W", md_path,
        title=(
            f"Benchmark: {benchmark}; method: {method}; "
            f"seeds: {sorted(seeds)}; limit: {limit}.\n"
            f"`W` here is the per-step MCTS simulation budget (paper "
            f"Table 15), NOT the per-leaf branching factor `W=3` from "
            f"paper §4.1."
        ),
    )
    print(f"[W-sweep] launched={launched} skipped={skipped} "
          f"jsonl={jsonl_path} md={md_path}")
    return {
        "dry_run": False,
        "launched": launched,
        "skipped": skipped,
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "summary": summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paper Table 15: W = sim-budget sweep")
    p.add_argument("--w-values", type=int, nargs="+", default=list(DEFAULT_W_VALUES),
                   help="W (per-step sim budget) values to sweep over (default: 4 8 16 32 64 128)")
    p.add_argument("--w-override", type=int, default=None,
                   help="Convenience: single W value (overrides --w-values).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK)
    p.add_argument("--method", type=str, default=DEFAULT_METHOD)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--config", type=str, default="default")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--jsonl", type=str, default=str(JSONL_PATH))
    p.add_argument("--md", type=str, default=str(MD_PATH))
    p.add_argument("--plan", type=str, default=str(PLAN_PATH))
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit; no eval launched.")
    p.add_argument("--force-no-gpu", action="store_true",
                   help="Bypass the auto-dry-run-on-no-GPU safety.")
    args = p.parse_args(argv)

    w_values = [args.w_override] if args.w_override is not None else list(args.w_values)

    auto_dry = False
    if not args.force_no_gpu and not args.dry_run:
        try:
            import torch
            if not torch.cuda.is_available():
                auto_dry = True
                print("[W-sweep] no CUDA device detected — auto-falling-back to --dry-run "
                      "(use --force-no-gpu to override).", flush=True)
        except Exception:
            auto_dry = True

    summary = run_sweep(
        w_values=w_values,
        seeds=args.seeds,
        benchmark=args.benchmark,
        method=args.method,
        limit=args.limit,
        config_name=args.config,
        device=args.device,
        model_dir=args.model_dir,
        jsonl_path=Path(args.jsonl),
        md_path=Path(args.md),
        plan_path=Path(args.plan),
        dry_run=args.dry_run or auto_dry,
    )
    print(json.dumps({"w_values": list(w_values), **{k: v for k, v in summary.items() if k != "summary"}}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
