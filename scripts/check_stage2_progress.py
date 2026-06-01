#!/usr/bin/env python3
"""Read-only Stage 2 PPO progress + post-S2 readiness gate (CPU-only).

Does NOT start GPU eval or touch the training process. Safe to run while
PID 37264 (or any in-flight Stage 2 job) is active.

Usage:
  python scripts/check_stage2_progress.py
  python scripts/check_stage2_progress.py --verify-final   # also load final ckpt if present
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESUME_NOTE = ROOT / "logs" / "STAGE2_RESUME_STATE.md"
DEFAULT_LOG = ROOT / "logs" / "stage2_paper_full_resume_20260517_204944.log"
FINAL_CKPT = ROOT / "artifacts" / "stage2_meta_value.pt"
INTER_CKPT = ROOT / "artifacts" / "stage2_meta_value.intermediate.pt"
STEP_RE = re.compile(r"stage2 step=(\d+)/(\d+)")


def _read_resume_pid() -> int | None:
    if not RESUME_NOTE.is_file():
        return None
    text = RESUME_NOTE.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"PID at snapshot\s*\|\s*\*\*(\d+)\*\*", text)
    return int(m.group(1)) if m else None


def _read_resume_log_path() -> Path:
    if RESUME_NOTE.is_file():
        text = RESUME_NOTE.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Stage 2 log file\s*\|\s*`([^`]+)`", text)
        if m:
            return Path(m.group(1))
    return DEFAULT_LOG


def _pid_alive(pid: int) -> bool | None:
    if sys.platform == "win32":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout or ""
        return str(pid) in out and "No tasks" not in out
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return None


def _tail_steps(log_path: Path) -> tuple[int | None, int | None]:
    if not log_path.is_file():
        return None, None
    last_step: int | None = None
    last_total: int | None = None
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                last_step, last_total = int(m.group(1)), int(m.group(2))
    return last_step, last_total


def _intermediate_step() -> int | None:
    if not INTER_CKPT.is_file():
        return None
    try:
        import torch

        sd = torch.load(INTER_CKPT, map_location="cpu", weights_only=False)
        meta = sd.get("training_meta") if isinstance(sd, dict) else None
        if isinstance(meta, dict) and meta.get("step") is not None:
            return int(meta["step"])
    except Exception:
        return None
    return None


def _verify_final_ckpt() -> dict:
    """Mirror ``run_post_stage2_pipeline.phase_verify_stage2`` (import-free copy)."""
    from scripts.run_post_stage2_pipeline import phase_verify_stage2

    ns = argparse.Namespace()
    return phase_verify_stage2(ns)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 progress (read-only)")
    ap.add_argument(
        "--verify-final",
        action="store_true",
        help="run final-ckpt verify when artifacts/stage2_meta_value.pt exists",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    log_path = _read_resume_log_path()
    pid = _read_resume_pid()
    step, total = _tail_steps(log_path)
    inter_step = _intermediate_step()
    pid_alive = _pid_alive(pid) if pid else None
    final_exists = FINAL_CKPT.is_file()

    ready_for_pipeline = bool(final_exists and step == total == 10000)
    verify_result: dict | None = None
    if args.verify_final and final_exists:
        verify_result = _verify_final_ckpt()
        ready_for_pipeline = verify_result.get("status") in ("PASS", "WARN")

    report = {
        "log_path": str(log_path),
        "log_exists": log_path.is_file(),
        "last_step": step,
        "total_steps": total,
        "intermediate_ckpt_step": inter_step,
        "pid": pid,
        "pid_alive": pid_alive,
        "final_ckpt_exists": final_exists,
        "ready_for_post_stage2_pipeline": ready_for_pipeline,
        "recommended_command_when_ready": (
            "python scripts/run_post_stage2_pipeline.py "
            "--device cuda:0 --table2-limit 50 "
            "--output-root results/post_stage2_May2026"
        ),
    }
    if verify_result is not None:
        report["final_ckpt_verify"] = verify_result

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== Stage 2 progress (read-only) ===")
        print(f"log:           {log_path} ({'found' if log_path.is_file() else 'MISSING'})")
        if step is not None:
            pct = 100.0 * step / total if total else 0.0
            print(f"last step:     {step}/{total} ({pct:.1f}%)")
        else:
            print("last step:     (no step lines in log yet)")
        if inter_step is not None:
            print(f"intermediate:  {INTER_CKPT.name} @ step {inter_step}")
        if pid:
            alive = {True: 'yes', False: 'no', None: 'unknown'}[pid_alive]
            print(f"training PID:  {pid} (alive: {alive})")
        print(f"final ckpt:    {FINAL_CKPT.name} ({'present' if final_exists else 'not yet'})")
        if verify_result:
            print(f"ckpt verify:   {verify_result.get('status')} - {verify_result.get('details')}")
        if final_exists and step is not None and total and step < total:
            print("NOTE:          final ckpt on disk may be STALE (log step < 10000).")
        print()
        if ready_for_pipeline:
            print("READY - run post-Stage-2 pipeline (GPU, after training process exits):")
            print(f"  {report['recommended_command_when_ready']}")
        elif step is not None and total and step < total:
            remaining = total - step
            eta_h = remaining * 90 / 3600
            print(f"IN PROGRESS - ~{remaining} steps left (~{eta_h:.1f} h @ 90 s/step)")
            print("Do NOT run GPU eval until step 10000 + fresh final ckpt written.")
        else:
            print("WAIT — training status unclear; see logs/STAGE2_RESUME_STATE.md")

    return 0 if ready_for_pipeline or (step is not None and total and step < total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
