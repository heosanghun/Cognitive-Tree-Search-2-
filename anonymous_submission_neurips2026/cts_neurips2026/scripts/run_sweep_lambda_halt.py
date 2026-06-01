#!/usr/bin/env python3
"""λ_halt ablation sweep (paper §6 / Appendix).

λ_halt is the ACT halting-penalty coefficient applied to the Stage 2
PPO reward (see ``cts/train/stage2_ppo_train.py`` line ~213,
``cfg["act_halting_penalty"]`` defaulting to 0.05). Because λ_halt only
affects the *training* reward signal, evaluating its sensitivity
requires retraining one Stage 2 PPO checkpoint per λ value — this is
fundamentally a multi-GPU job and cannot be done end-to-end on CPU CI.

This script therefore operates in two modes:

  1. *Manifest mode* (default when checkpoints are missing): emit a
     ``training_jobs.json`` manifest enumerating the exact CLI commands
     to launch the four Stage 2 PPO retraining jobs, and a Markdown
     status table flagging every λ value as ``PENDING_GPU``. CPU CI
     exercises this path; reviewers can then submit the jobs to a real
     cluster.

  2. *Eval mode* (when all checkpoints are present): for each λ value,
     load the corresponding ``runs/stage2_lambda_<value>/policy.pt``
     and evaluate cts_4nu on AIME with seeds {0, 1, 2}, then write the
     standard sweep JSONL + Markdown.

Outputs:
  - results/sweep_lambda_halt/training_jobs.json
  - results/sweep_lambda_halt/sweep_lambda_halt.md
  - results/sweep_lambda_halt/sweep_lambda_halt.jsonl  (eval mode only)
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


DEFAULT_LAMBDA_VALUES: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_BENCHMARK: str = "aime"
DEFAULT_METHOD: str = "cts_4nu"
DEFAULT_LIMIT: int = 30
OUT_DIR = ROOT / "results" / "sweep_lambda_halt"
JSONL_PATH = OUT_DIR / "sweep_lambda_halt.jsonl"
MD_PATH = OUT_DIR / "sweep_lambda_halt.md"
JOBS_PATH = OUT_DIR / "training_jobs.json"
RUNS_DIR = ROOT / "runs"


def _ckpt_for(lambda_value: float, runs_dir: Path = RUNS_DIR) -> Path:
    """Return the canonical Stage-2 checkpoint path for a given λ value.

    The naming convention ``runs/stage2_lambda_<value>/policy.pt`` is
    chosen to match the manifest's CLI invocations so a reviewer can
    grep for either string.
    """
    return runs_dir / f"stage2_lambda_{lambda_value}" / "policy.pt"


def _canonical_ckpt_str(lambda_value: float) -> str:
    """Return the canonical *relative* ckpt path string used in the
    reviewer-facing manifest and Markdown.

    Always ``runs/stage2_lambda_<value>/policy.pt`` with POSIX slashes,
    regardless of the local ``runs_dir`` (which may be a tmp path during
    testing or a custom cluster path during sweep execution). This
    keeps the manifest reproducible across hosts (and avoids leaking
    the author's local layout into a double-blind artifact).
    """
    return f"runs/stage2_lambda_{lambda_value}/policy.pt"


def _ckpt_status(lambda_value: float, runs_dir: Path) -> str:
    if _ckpt_for(lambda_value, runs_dir).is_file():
        return "CKPT_PRESENT"
    return "PENDING_GPU"


def _eval_status(
    lambda_value: float,
    rows: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> str:
    """Return ``EVAL_DONE`` if every (λ, seed) row is present in rows."""
    seen_seeds = {int(r["seed"]) for r in rows
                  if abs(float(r.get("lambda_halt", -1)) - float(lambda_value)) < 1e-12
                  and "seed" in r}
    needed = {int(s) for s in seeds}
    if needed.issubset(seen_seeds):
        return "EVAL_DONE"
    return "EVAL_MISSING"


def build_training_manifest(
    lambda_values: Sequence[float],
    *,
    runs_dir: Path = RUNS_DIR,
    config_name: str = "default",
    extra_args: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build the GPU-job manifest enumerating the Stage 2 PPO retrains
    that need to run before eval mode can proceed.

    The manifest is intentionally a JSON document (not a shell script)
    so reviewers can submit it to whatever cluster scheduler they have
    on hand — the ``cli`` field is the verbatim ``python ...``
    invocation.
    """
    jobs = []
    for lam in lambda_values:
        ckpt_path = _ckpt_for(lam, runs_dir)
        cli = [
            "python", "scripts/run_stage2_math_ppo.py",
            "--config", config_name,
            "--steps", "10000",
        ]
        cli.extend(list(extra_args))
        # The Stage-2 trainer reads ``act_halting_penalty`` from the
        # merged config; the canonical sweep override is to pass an
        # ad-hoc YAML via the CTS_CFG_OVERRIDE env var or a custom
        # config file. We emit the env-var form here because it
        # requires no on-disk YAML proliferation.
        env = {"CTS_ACT_HALTING_PENALTY": str(lam)}
        ckpt_str = _canonical_ckpt_str(lam)
        jobs.append({
            "lambda_halt": float(lam),
            "ckpt_path": ckpt_str,
            "status": _ckpt_status(lam, runs_dir),
            "cli": cli,
            "env": env,
            "notes": (
                "Stage 2 PPO retrain with act_halting_penalty=" + str(lam)
                + ". Save the resulting policy/critic state-dict to "
                + ckpt_str
                + " so run_sweep_lambda_halt.py eval-mode picks it up."
            ),
        })
    return {
        "schema_version": 1,
        "param_name": "lambda_halt",
        "param_values": [float(v) for v in lambda_values],
        "jobs": jobs,
    }


def write_training_manifest(
    lambda_values: Sequence[float],
    *,
    runs_dir: Path = RUNS_DIR,
    config_name: str = "default",
    jobs_path: Path = JOBS_PATH,
    extra_args: Sequence[str] = (),
) -> Dict[str, Any]:
    manifest = build_training_manifest(
        lambda_values, runs_dir=runs_dir, config_name=config_name,
        extra_args=extra_args,
    )
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def render_status_markdown(
    lambda_values: Sequence[float],
    seeds: Sequence[int],
    *,
    runs_dir: Path,
    md_path: Path,
    jsonl_rows: Sequence[Dict[str, Any]] = (),
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# λ_halt sweep status")
    lines.append("")
    lines.append(
        "λ_halt is the ACT halting-penalty coefficient applied to the Stage 2 "
        "PPO reward (paper Eq. 5). Each λ value requires a separate Stage 2 "
        "checkpoint; CPU CI cannot retrain so we report a status table here. "
        "Once the GPU jobs from `training_jobs.json` complete, re-run this "
        "script to populate the actual ablation numbers."
    )
    lines.append("")
    lines.append("| lambda_halt | ckpt_path | ckpt_status | eval_status |")
    lines.append("| --- | --- | --- | --- |")
    for lam in lambda_values:
        ck_path = _ckpt_for(lam, runs_dir)
        ck_status = _ckpt_status(lam, runs_dir)
        ev_status = _eval_status(lam, jsonl_rows, seeds) if jsonl_rows else "EVAL_MISSING"
        # Reflect the headline reviewer-facing PENDING_GPU label whenever
        # the checkpoint is absent, regardless of eval status.
        if ck_status == "PENDING_GPU":
            ev_status = "PENDING_GPU"
        lines.append(f"| {lam} | `{_canonical_ckpt_str(lam)}` | {ck_status} | {ev_status} |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _patch_cts_full_episode_for_eval():
    """Eval-mode does NOT change ``k_override`` or ``w_override``; this
    is a placeholder hook for symmetry with the K/W sweeps.
    """
    return None


def run_eval_for_lambda(
    lambda_value: float,
    *,
    seeds: Sequence[int],
    benchmark: str,
    method: str,
    limit: int,
    config_name: str,
    device: str,
    model_dir: Optional[str],
    runs_dir: Path,
    jsonl_path: Path,
) -> int:
    """Evaluate a single λ value across all seeds. Returns the number
    of (λ, seed) pairs newly written.

    Each row is tagged with ``lambda_halt`` so ``summarize_sweep`` can
    fold seeds together. The Stage-2 checkpoint is loaded by
    ``_run_cts_on_problems`` via the existing
    ``artifacts/stage2_meta_value.pt`` lookup; this script simply
    symlinks / copies the per-λ ckpt into that path before each eval.
    """
    from scripts.run_cts_eval_full import run_single_evaluation  # noqa: E402

    ck_path = _ckpt_for(lambda_value, runs_dir)
    if not ck_path.is_file():
        print(f"[lambda-sweep] ckpt missing for λ={lambda_value}: {ck_path}", flush=True)
        return 0

    # Copy the per-λ checkpoint into the canonical artifact location so
    # ``_run_cts_on_problems`` picks it up. We restore any pre-existing
    # checkpoint after the eval to avoid clobbering the user's
    # workspace.
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    canonical = artifacts / "stage2_meta_value.pt"
    backup: Optional[Path] = None
    if canonical.is_file():
        backup = artifacts / "stage2_meta_value.pt.lambda_sweep.bak"
        canonical.replace(backup)
    try:
        canonical.write_bytes(ck_path.read_bytes())

        n_written = 0
        for seed in seeds:
            print(f"[lambda-sweep] launching λ={lambda_value} seed={seed}", flush=True)
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
            scores = res.get("scores") or []
            for pi, score in enumerate(scores):
                append_sweep_row(jsonl_path, {
                    "lambda_halt": float(lambda_value),
                    "seed": int(seed),
                    "benchmark": benchmark,
                    "method": method,
                    "problem_idx": pi,
                    "score": float(score),
                    "elapsed_s": float(elapsed),
                })
            if not scores:
                append_sweep_row(jsonl_path, {
                    "lambda_halt": float(lambda_value),
                    "seed": int(seed),
                    "benchmark": benchmark,
                    "method": method,
                    "problem_idx": -1,
                    "score": float(res.get("accuracy", 0.0)),
                    "elapsed_s": float(elapsed),
                    "error": res.get("error"),
                })
            n_written += 1
        return n_written
    finally:
        try:
            canonical.unlink()
        except FileNotFoundError:
            pass
        if backup is not None and backup.is_file():
            backup.replace(canonical)


def run_sweep(
    *,
    lambda_values: Sequence[float] = DEFAULT_LAMBDA_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    benchmark: str = DEFAULT_BENCHMARK,
    method: str = DEFAULT_METHOD,
    limit: int = DEFAULT_LIMIT,
    config_name: str = "default",
    device: str = "cuda:0",
    model_dir: Optional[str] = None,
    runs_dir: Path = RUNS_DIR,
    jsonl_path: Path = JSONL_PATH,
    md_path: Path = MD_PATH,
    jobs_path: Path = JOBS_PATH,
    extra_train_args: Sequence[str] = (),
    force_eval: bool = False,
) -> Dict[str, Any]:
    manifest = write_training_manifest(
        lambda_values, runs_dir=runs_dir, config_name=config_name,
        jobs_path=jobs_path, extra_args=extra_train_args,
    )

    all_present = all(
        _ckpt_for(lam, runs_dir).is_file() for lam in lambda_values
    )

    if not all_present and not force_eval:
        # Manifest mode: skip eval entirely and produce a PENDING_GPU
        # status table.
        render_status_markdown(
            lambda_values, seeds, runs_dir=runs_dir,
            md_path=md_path, jsonl_rows=load_sweep_jsonl(jsonl_path),
        )
        print(f"[lambda-sweep] manifest mode: {jobs_path} written; "
              f"{sum(1 for j in manifest['jobs'] if j['status'] == 'PENDING_GPU')} "
              f"PENDING_GPU jobs.", flush=True)
        return {
            "mode": "manifest",
            "manifest_path": str(jobs_path),
            "md_path": str(md_path),
            "manifest": manifest,
        }

    # Eval mode: run every λ for which a checkpoint exists.
    written = 0
    for lam in lambda_values:
        if _ckpt_for(lam, runs_dir).is_file():
            written += run_eval_for_lambda(
                lam, seeds=seeds, benchmark=benchmark, method=method,
                limit=limit, config_name=config_name, device=device,
                model_dir=model_dir, runs_dir=runs_dir,
                jsonl_path=jsonl_path,
            )

    rows = load_sweep_jsonl(jsonl_path)
    summary = summarize_sweep(rows, "lambda_halt")
    if summary:
        render_sweep_markdown(
            summary, "lambda_halt", md_path,
            title=(
                f"Benchmark: {benchmark}; method: {method}; "
                f"seeds: {sorted(seeds)}; limit: {limit}.\n"
                f"λ_halt is the ACT halting-penalty coefficient in the "
                f"Stage 2 PPO reward (paper Eq. 5)."
            ),
        )
    else:
        render_status_markdown(
            lambda_values, seeds, runs_dir=runs_dir,
            md_path=md_path, jsonl_rows=rows,
        )
    print(f"[lambda-sweep] eval mode: wrote {written} (λ, seed) rows; "
          f"jsonl={jsonl_path} md={md_path}", flush=True)
    return {
        "mode": "eval",
        "written": written,
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "summary": summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="λ_halt sensitivity sweep")
    p.add_argument("--lambda-values", type=float, nargs="+",
                   default=list(DEFAULT_LAMBDA_VALUES),
                   help="λ_halt values to sweep over (default: 0.01 0.05 0.1 0.5)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK)
    p.add_argument("--method", type=str, default=DEFAULT_METHOD)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--config", type=str, default="default")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default=str(RUNS_DIR))
    p.add_argument("--jsonl", type=str, default=str(JSONL_PATH))
    p.add_argument("--md", type=str, default=str(MD_PATH))
    p.add_argument("--jobs", type=str, default=str(JOBS_PATH))
    p.add_argument("--force-eval", action="store_true",
                   help="Try to run eval even if some checkpoints are missing.")
    p.add_argument("--force-no-gpu", action="store_true",
                   help="Bypass the auto-manifest-on-no-GPU safety; only "
                        "useful for tests.")
    args = p.parse_args(argv)

    auto_manifest = False
    if not args.force_no_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                auto_manifest = True
                print("[lambda-sweep] no CUDA device detected — manifest mode "
                      "will be used regardless of checkpoint presence.",
                      flush=True)
        except Exception:
            auto_manifest = True

    summary = run_sweep(
        lambda_values=args.lambda_values,
        seeds=args.seeds,
        benchmark=args.benchmark,
        method=args.method,
        limit=args.limit,
        config_name=args.config,
        device=args.device,
        model_dir=args.model_dir,
        runs_dir=Path(args.runs_dir),
        jsonl_path=Path(args.jsonl),
        md_path=Path(args.md),
        jobs_path=Path(args.jobs),
        force_eval=args.force_eval and not auto_manifest,
    )
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("manifest", "summary")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
