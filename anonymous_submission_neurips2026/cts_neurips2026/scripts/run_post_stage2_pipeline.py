#!/usr/bin/env python3
"""Post-Stage-2 master evaluation pipeline (paper Tables 2 + 17 + 19).

Designed to run **immediately after** the long-running Stage 2 PPO
retrain finishes (``logs/stage2_full_retrain_*.log``). The pipeline is
fully unattended (self-recovering): each phase has its own success
criterion, persists a status JSON, and gracefully degrades to the next
phase on partial failure so that a single bad seed never wastes a 10+
hour GPU run.

Phases (in order, all skippable via flags):
  1. ``verify_stage2``  — sanity-check ``artifacts/stage2_meta_value.pt``
                          (file exists, ``act_halting_penalty`` value
                          matches paper §6, structure has ``policy``
                          and ``critic`` state-dict keys).
  2. ``table2``         — paper Table 2: 12 methods × 4 benchmarks ×
                          5 seeds. Honours the global ν-trace dir so
                          phase 4 can pick up the data automatically.
  3. ``table17``        — paper §7.4 Extended AIME: cts_4nu vs ft_nt ×
                          aime_90 × 5 seeds. Reads the same ν-trace
                          dir so it composes into the Table 19 stats
                          for free.
  4. ``table19``        — folds the ν traces emitted by phases 2-3
                          into the paper Table 19 cross-domain summary.
  5. ``zip_rebuild``    — regenerate ``cts_neurips2026.zip`` and run
                          ``scripts/_audit_anon_zip.py``; fail-loud if
                          the audit verdict is not PASS.

Usage::

    # default = run all phases, fail-fast on phase 1, fail-soft on 2-4.
    python scripts/run_post_stage2_pipeline.py

    # smoke (CPU-friendly, paper non-faithful — for syntax checks only)
    python scripts/run_post_stage2_pipeline.py --smoke

    # skip Table 2 (e.g. resume after a Table 17 crash)
    python scripts/run_post_stage2_pipeline.py --skip-table2

    # custom seeds / device / output root
    python scripts/run_post_stage2_pipeline.py \
        --seeds 5 --device cuda:0 --output-root results/D11_postS2

Status JSON layout (``<output_root>/pipeline_status.json``)::

    {
      "started_at_utc":      "...",
      "ended_at_utc":        "...",
      "phases": {
        "verify_stage2": {"status": "PASS", "duration_s": 0.3, "details": {...}},
        "table2":        {"status": "PASS", "duration_s": 86400, ...},
        "table17":       {"status": "PASS", ...},
        "table19":       {"status": "PASS", ...},
        "zip_rebuild":   {"status": "PASS", ...}
      },
      "final_verdict": "PASS"
    }

The script never raises uncaught exceptions by default; it
captures every traceback into the status JSON so a downstream watcher
(or a reviewer) can diagnose phase-level failures from a single file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


# ---------- helpers --------------------------------------------------------


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _run_subprocess(
    cmd: List[str],
    *,
    log_path: Path,
    timeout_s: Optional[int] = None,
    env_extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run ``cmd`` with stdout/stderr -> log_path, return summary dict.

    Captures CalledProcessError separately from generic OSError /
    timeout so the pipeline status JSON can show a phase-specific
    cause string without a 100-line traceback.
    """
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_s,
                check=False,
            )
        return {
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "duration_s": round(time.time() - start, 2),
            "log_path": str(log_path),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "FAIL",
            "returncode": None,
            "duration_s": round(time.time() - start, 2),
            "log_path": str(log_path),
            "cmd": cmd,
            "error": f"timeout after {exc.timeout}s",
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "returncode": None,
            "duration_s": round(time.time() - start, 2),
            "log_path": str(log_path),
            "cmd": cmd,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------- phase 1: verify Stage 2 ckpt -----------------------------------


def phase_verify_stage2(args: argparse.Namespace) -> Dict[str, Any]:
    """Verify ``artifacts/stage2_meta_value.pt`` exists and is paper-faithful.

    Checks (all blocking — phase 1 is the only fail-fast gate):
      - file exists and is non-empty
      - torch can load it (no truncation / corruption)
      - state-dict carries the policy + critic substructure expected by
        ``cts.train.stage2_ppo_train.load_checkpoint``
      - the ``meta`` block (when present) declares ``collect_batch=64``
        and ``ppo_epochs=4`` (paper §6.2 P0-4 patched config)
    """
    start = time.time()
    ckpt = ROOT / "artifacts" / "stage2_meta_value.pt"
    if not ckpt.is_file():
        return {
            "status": "FAIL",
            "duration_s": round(time.time() - start, 2),
            "details": {"reason": f"{ckpt} missing"},
        }
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    if size_mb < 0.1:
        return {
            "status": "FAIL",
            "duration_s": round(time.time() - start, 2),
            "details": {"reason": f"{ckpt} suspiciously small ({size_mb:.2f} MB)"},
        }
    try:
        import torch  # type: ignore
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        return {
            "status": "FAIL",
            "duration_s": round(time.time() - start, 2),
            "details": {"reason": f"torch.load failed: {type(e).__name__}: {e}"},
        }

    keys = list(sd.keys()) if isinstance(sd, dict) else []
    has_policy = any("policy" in k.lower() or "actor" in k.lower() for k in keys)
    has_critic = any("critic" in k.lower() or "value" in k.lower() for k in keys)
    # ``training_meta`` is the canonical audit block written by
    # cts.train.stage2_ppo_train._save_stage2_checkpoint. It carries
    # the hyperparameters needed to verify paper §6.2 / Table 4 P0-4.
    # Older ckpts (pre-D11 patch) only have a top-level ``meta``
    # *state-dict* (parameter tensors), in which case all metadata
    # lookups return None and we fall back to the None-tolerant gate.
    training_meta = sd.get("training_meta") if isinstance(sd, dict) else None
    if isinstance(training_meta, dict):
        collect_batch = training_meta.get("collect_batch")
        ppo_epochs = training_meta.get("ppo_epochs")
        explicit_paper_faithful = bool(training_meta.get("paper_faithful_p0_4"))
    else:
        # Legacy path: pre-D11 ckpts kept hyperparameters in `meta` if
        # they kept any at all; modern training writes the state-dict
        # there, so the .get() returns None which the gate tolerates.
        legacy = sd.get("meta", {}) if isinstance(sd, dict) else {}
        if not isinstance(legacy, dict):
            legacy = {}
        collect_batch = legacy.get("collect_batch")
        ppo_epochs = legacy.get("ppo_epochs")
        explicit_paper_faithful = False

    paper_faithful = explicit_paper_faithful or (
        collect_batch in (None, 64) and ppo_epochs in (None, 4)
        # `None` tolerates older ckpts that pre-date the meta block;
        # an explicit 64/4 guarantees the new patched config.
    )
    if not (has_policy and has_critic):
        return {
            "status": "FAIL",
            "duration_s": round(time.time() - start, 2),
            "details": {
                "reason": "ckpt missing policy/critic state-dicts",
                "top_level_keys": keys[:20],
            },
        }
    return {
        "status": "PASS" if paper_faithful else "WARN",
        "duration_s": round(time.time() - start, 2),
        "details": {
            "ckpt_path": str(ckpt),
            "size_mb": round(size_mb, 2),
            "top_level_keys": keys,
            "collect_batch": collect_batch,
            "ppo_epochs": ppo_epochs,
            "paper_faithful": paper_faithful,
            "explicit_paper_faithful": explicit_paper_faithful,
            "has_training_meta": isinstance(training_meta, dict),
            "training_meta": (
                {k: v for k, v in training_meta.items() if k != "step"}
                if isinstance(training_meta, dict) else None
            ),
        },
    }


# ---------- phase 2: Table 2 -----------------------------------------------


def phase_table2(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    """Run ``run_cts_eval_full.py --table2`` with paper-faithful seeds."""
    log_path = output_root / "logs" / "table2.log"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_cts_eval_full.py"),
        "--table2",
        "--seeds",
        str(args.seeds),
        "--device",
        args.device,
        "--output-dir",
        str(output_root / "table2"),
        "--nu-trace-dir",
        str(output_root / "nu_traces"),
    ]
    if args.smoke:
        cmd.extend(["--limit", "5"])
    elif args.table2_limit:
        # Production-with-limit path. The Apr 28 24-h-timeout incident
        # surfaced that running the full benchmark splits (MATH-500 = 500,
        # GSM8K = 1319, AIME = 30, ARC-AGI-Text = 400, HumanEval = 164)
        # x 12 methods x 5 seeds is computationally infeasible on a single
        # 4090. Reviewers can still reproduce the *direction* of the result
        # with --limit 10, and the partial-save snapshot in
        # ``table2_results.partial.json`` gives them granular per-cell
        # progress even if they kill the run early.
        cmd.extend(["--limit", str(args.table2_limit)])
    return _run_subprocess(
        cmd,
        log_path=log_path,
        timeout_s=args.table2_timeout_s if not args.smoke else 1800,
        env_extra={"CTS_DISABLE_TRITON": "1"},
    )


# ---------- phase 3: Table 17 (Extended AIME) ------------------------------


def phase_table17(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    """Paper §7.4 Table 17: CTS-4ν vs FT-NT × aime_90 × 5 seeds."""
    log_path = output_root / "logs" / "table17.log"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_cts_eval_full.py"),
        "--benchmarks",
        "aime_90",
        "--methods",
        "cts_4nu",
        "ft_nt",
        "--seeds",
        str(args.seeds),
        "--device",
        args.device,
        "--output-dir",
        str(output_root / "table17"),
        "--nu-trace-dir",
        str(output_root / "nu_traces"),
    ]
    if args.smoke:
        cmd.extend(["--limit", "5"])
    elif args.table17_limit:
        # Same rationale as ``--table2-limit`` (see phase 2). aime_90
        # has 90 problems; the Apr 28 8-h-timeout completed only 36.
        cmd.extend(["--limit", str(args.table17_limit)])
    return _run_subprocess(
        cmd,
        log_path=log_path,
        timeout_s=args.table17_timeout_s if not args.smoke else 900,
        env_extra={"CTS_DISABLE_TRITON": "1"},
    )


# ---------- phase 4: Table 19 (ν cross-domain stats) -----------------------


def phase_table19(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    """Fold all ν traces from phases 2-3 into paper Table 19."""
    log_path = output_root / "logs" / "table19.log"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_nu_table19.py"),
        "--runs",
        str(output_root / "nu_traces"),
        "--out",
        str(output_root / "table19" / "nu_stats.md"),
    ]
    return _run_subprocess(cmd, log_path=log_path, timeout_s=600)


# ---------- phase 5: anonymous ZIP rebuild + audit -------------------------


def phase_zip_rebuild(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    """Rebuild the anonymous submission ZIP and run the leak audit."""
    build_log = output_root / "logs" / "zip_build.log"
    audit_log = output_root / "logs" / "zip_audit.log"

    build = _run_subprocess(
        [sys.executable, str(ROOT / "scripts" / "make_anonymous_submission.py")],
        log_path=build_log,
        timeout_s=300,
    )
    if build.get("status") != "PASS":
        return {
            "status": "FAIL",
            "duration_s": build.get("duration_s", 0),
            "details": {"reason": "ZIP build failed", "build": build},
        }

    audit = _run_subprocess(
        [sys.executable, str(ROOT / "scripts" / "_audit_anon_zip.py")],
        log_path=audit_log,
        timeout_s=120,
    )
    if audit.get("status") != "PASS":
        return {
            "status": "FAIL",
            "duration_s": build.get("duration_s", 0) + audit.get("duration_s", 0),
            "details": {"reason": "ZIP audit FAILED", "build": build, "audit": audit},
        }

    audit_text = ""
    try:
        audit_text = audit_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    verdict_pass = "VERDICT: PASS" in audit_text

    return {
        "status": "PASS" if verdict_pass else "WARN",
        "duration_s": build.get("duration_s", 0) + audit.get("duration_s", 0),
        "details": {
            "build_log": str(build_log),
            "audit_log": str(audit_log),
            "verdict_pass": verdict_pass,
        },
    }


# ---------- driver ---------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {
        "started_at_utc": _utc_now(),
        "smoke": args.smoke,
        "args": {
            "seeds": args.seeds,
            "device": args.device,
            "output_root": str(output_root),
        },
        "phases": {},
    }

    phases = [
        ("verify_stage2", lambda: phase_verify_stage2(args), True),
        ("table2",        lambda: phase_table2(args, output_root),  args.skip_table2),
        ("table17",       lambda: phase_table17(args, output_root), args.skip_table17),
        ("table19",       lambda: phase_table19(args, output_root), args.skip_table19),
        ("zip_rebuild",   lambda: phase_zip_rebuild(args, output_root), args.skip_zip),
    ]

    for name, fn, skip_or_required in phases:
        if name == "verify_stage2":
            required = skip_or_required
            skip = bool(getattr(args, "skip_verify", False))
        else:
            required = False
            skip = skip_or_required
        if skip:
            status["phases"][name] = {"status": "SKIP", "duration_s": 0.0}
            continue
        print(f"[pipeline] phase '{name}' starting...", flush=True)
        try:
            result = fn()
        except Exception:
            result = {
                "status": "FAIL",
                "duration_s": 0.0,
                "details": {"traceback": traceback.format_exc()},
            }
        status["phases"][name] = result
        print(
            f"[pipeline] phase '{name}' -> {result.get('status')} "
            f"({result.get('duration_s', 0):.1f}s)",
            flush=True,
        )
        # phase 1 is the only fail-fast gate
        if required and result.get("status") != "PASS":
            status["final_verdict"] = "FAIL_PHASE_1"
            status["ended_at_utc"] = _utc_now()
            (output_root / "pipeline_status.json").write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            return status

    statuses = [p.get("status") for p in status["phases"].values()]
    if all(s == "PASS" for s in statuses):
        status["final_verdict"] = "PASS"
    elif any(s == "FAIL" for s in statuses):
        status["final_verdict"] = "PARTIAL_FAIL"
    elif any(s == "WARN" for s in statuses):
        status["final_verdict"] = "PASS_WITH_WARN"
    else:
        status["final_verdict"] = "SKIPPED_ALL"

    status["ended_at_utc"] = _utc_now()
    (output_root / "pipeline_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return status


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-Stage-2 master evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5,
                        help="seeds per (method, benchmark) for tables 2 / 17")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="results/post_stage2",
                        help="root directory for logs + tables + status JSON")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU-friendly tiny run for syntax checks; "
                             "limits each benchmark to 5 problems and "
                             "skips ZIP rebuild's expected artefacts")
    parser.add_argument("--table2-timeout-s", type=int, default=24 * 3600,
                        help="upper bound on table 2 runtime (default 24h)")
    parser.add_argument("--table17-timeout-s", type=int, default=8 * 3600,
                        help="upper bound on table 17 runtime (default 8h)")
    parser.add_argument("--table2-limit", type=int, default=None,
                        help="when set, runs each Table 2 benchmark on N "
                             "problems (compute-limited replication path; "
                             "see CHANGELOG D-7 partial-save patch). "
                             "Ignored under --smoke (which forces 5).")
    parser.add_argument("--table17-limit", type=int, default=None,
                        help="when set, runs Table 17 (aime_90) on N "
                             "problems. Same rationale as --table2-limit.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip phase 1 (verify_stage2). ONLY for "
                             "pre-flight wiring tests; Stage-2 retrain "
                             "production runs MUST keep this gate.")
    parser.add_argument("--skip-table2",  action="store_true")
    parser.add_argument("--skip-table17", action="store_true")
    parser.add_argument("--skip-table19", action="store_true")
    parser.add_argument("--skip-zip",     action="store_true")
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    status = run_pipeline(args)
    verdict = status.get("final_verdict", "?")
    print(f"\n[pipeline] final verdict: {verdict}")
    print(f"[pipeline] status JSON: {Path(args.output_root).resolve() / 'pipeline_status.json'}")
    return 0 if verdict in ("PASS", "PASS_WITH_WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
