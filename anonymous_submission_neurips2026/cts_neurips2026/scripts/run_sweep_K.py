#!/usr/bin/env python3
"""Paper Table 13: K = MCTS top-K children sensitivity sweep.

K here is the per-leaf children-expansion count in the PUCT tree, NOT
the latent-token bottleneck width K=64 from paper §4.2 (the latter is
unaffected by this sweep — see ``cts/mcts/cts_episode.py`` docstring on
``k_override``).

Defaults reproduce paper Table 13:
  - K values: {2, 3, 4, 5, 6, 8}
  - benchmark: AIME (data/aime/test.jsonl)
  - method:    cts_4nu
  - seeds:     {0, 1, 2}
  - limit:     30 (full AIME 2026 test set)

CPU-only invocations:
    python scripts/run_sweep_K.py --dry-run --k-values 2 3 --seeds 0

The script auto-falls-back to ``--dry-run`` when no CUDA device is
available so CI runs do not burn GPU-grade wall clock by accident; pass
``--force-no-gpu`` to override (used by the monkey-patched regression
test).

Outputs:
  - results/sweep_K/sweep_K.jsonl  (one row per (K, seed, problem))
  - results/sweep_K/sweep_K.md     (mean ± 95% bootstrap CI per K)
  - results/sweep_K/sweep_K_plan.txt (planned grid; written even on dry-run)
"""

from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_K_VALUES: Tuple[int, ...] = (2, 3, 4, 5, 6, 8)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_BENCHMARK: str = "aime"
DEFAULT_METHOD: str = "cts_4nu"
DEFAULT_LIMIT: int = 30
OUT_DIR = ROOT / "results" / "sweep_K"
JSONL_PATH = OUT_DIR / "sweep_K.jsonl"
MD_PATH = OUT_DIR / "sweep_K.md"
PLAN_PATH = OUT_DIR / "sweep_K_plan.txt"


def _completed_pairs(jsonl_path: Path, param_name: str) -> set:
    """Return {(K, seed)} pairs already present in JSONL for idempotent re-runs."""
    rows = load_sweep_jsonl(jsonl_path)
    done = set()
    for r in rows:
        if param_name in r and "seed" in r:
            try:
                done.add((int(r[param_name]), int(r["seed"])))
            except (TypeError, ValueError):
                continue
    return done


def _patch_cts_full_episode_with_k_override(k_value: int):
    """Monkey-patch cts.mcts.cts_episode.cts_full_episode to thread
    ``k_override=k_value`` through every call site (including the
    deferred ``from cts.mcts.cts_episode import cts_full_episode``
    re-imports inside ``_run_cts_on_problems``). Returns the original
    callable so the caller can restore it via ``_restore``.
    """
    from cts.mcts import cts_episode as _ep
    real = _ep.cts_full_episode

    def _wrapped(*args, **kwargs):
        kwargs.setdefault("k_override", k_value)
        return real(*args, **kwargs)

    _ep.cts_full_episode = _wrapped  # type: ignore[assignment]
    return real


def _restore_cts_full_episode(real_fn) -> None:
    from cts.mcts import cts_episode as _ep
    _ep.cts_full_episode = real_fn  # type: ignore[assignment]


def plan_sweep(
    k_values: Sequence[int],
    seeds: Sequence[int],
    benchmark: str,
) -> List[Tuple[int, int, str]]:
    """Return the full planned grid as ``(K, seed, benchmark)`` tuples."""
    return dry_run_grid(k_values, seeds, [benchmark])  # type: ignore[return-value]


def write_plan(
    grid: Sequence[Tuple[int, int, str]],
    plan_path: Path,
) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        f.write(f"# K-sweep plan ({len(grid)} jobs)\n")
        f.write("# format: K seed benchmark\n")
        for k, s, b in grid:
            f.write(f"{k}\t{s}\t{b}\n")


def run_sweep(
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
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
    """Run the K sweep. Returns a small summary dict.

    Idempotent: any (K, seed) pair already present in ``jsonl_path``
    is skipped. The planned grid is always written to ``plan_path``.
    On ``dry_run=True``, no eval is launched and an empty summary is
    returned (the JSONL file is left untouched).
    """
    grid = plan_sweep(k_values, seeds, benchmark)
    write_plan(grid, plan_path)

    if dry_run:
        print(f"[dry-run] planned {len(grid)} (K, seed) eval jobs:")
        for k, s, b in grid:
            print(f"  K={k} seed={s} bench={b} method={method} limit={limit}")
        print(f"[dry-run] plan written to {plan_path}")
        print(f"[dry-run] no JSONL written; exiting before eval launch.")
        return {"dry_run": True, "planned_jobs": len(grid)}

    done = _completed_pairs(jsonl_path, "K")

    from scripts.run_cts_eval_full import run_single_evaluation  # noqa: E402

    launched = 0
    skipped = 0
    for k_val, seed, _bench in grid:
        if (k_val, seed) in done:
            skipped += 1
            continue
        print(f"[K-sweep] launching K={k_val} seed={seed} bench={benchmark} method={method}")
        real = _patch_cts_full_episode_with_k_override(k_val)
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
                "K": int(k_val),
                "seed": int(seed),
                "benchmark": benchmark,
                "method": method,
                "problem_idx": pi,
                "score": float(score),
                "elapsed_s": float(elapsed),
            })
        if not scores:
            append_sweep_row(jsonl_path, {
                "K": int(k_val),
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
    summary = summarize_sweep(rows, "K")
    render_sweep_markdown(
        summary, "K", md_path,
        title=(
            f"Benchmark: {benchmark}; method: {method}; "
            f"seeds: {sorted(seeds)}; limit: {limit}.\n"
            f"`K` here is the MCTS per-leaf children-expansion count "
            f"(paper Table 13), NOT the latent-token bottleneck width "
            f"`K=64` from paper §4.2."
        ),
    )
    print(f"[K-sweep] launched={launched} skipped={skipped} "
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
    p = argparse.ArgumentParser(description="Paper Table 13: K = top-K children sweep")
    p.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                   help="K (top-K children) values to sweep over (default: 2 3 4 5 6 8)")
    p.add_argument("--k-override", type=int, default=None,
                   help="Convenience: single K value (overrides --k-values).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="Random seeds (default: 0 1 2)")
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK,
                   help="Benchmark name (default: aime)")
    p.add_argument("--method", type=str, default=DEFAULT_METHOD,
                   help="CTS method dispatcher (default: cts_4nu)")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Per-benchmark problem cap (default: 30 = full AIME 2026 test set)")
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

    k_values = [args.k_override] if args.k_override is not None else list(args.k_values)

    auto_dry = False
    if not args.force_no_gpu and not args.dry_run:
        try:
            import torch
            if not torch.cuda.is_available():
                auto_dry = True
                print("[K-sweep] no CUDA device detected — auto-falling-back to --dry-run "
                      "(use --force-no-gpu to override).", flush=True)
        except Exception:
            auto_dry = True

    summary = run_sweep(
        k_values=k_values,
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
    print(json.dumps({"k_values": list(k_values), **{k: v for k, v in summary.items() if k != "summary"}}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
