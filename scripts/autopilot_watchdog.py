#!/usr/bin/env python3
"""Watchdog: keep post-S2 autopilot / Wave 2 eval alive without user intervention.

Polls log + process state; restarts autopilot when eval stalls or crashes mid-Wave-2.
Reads policy from configs/autopilot_autonomous.json.

Usage:
  python scripts/autopilot_watchdog.py --watch
  python scripts/autopilot_watchdog.py --once
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "configs" / "autopilot_autonomous.json"
STATUS_PATH = ROOT / "results" / "post_s2_autopilot" / "autopilot_status.json"
WAVE2_LOG = ROOT / "results" / "post_s2_autopilot" / "logs" / "wave2.log"
WATCHDOG_LOG = ROOT / "results" / "post_s2_autopilot" / "logs" / "watchdog.log"
LAUNCHER = ROOT / "scripts" / "start_post_s2_autopilot.ps1"

PROB_RE = re.compile(r"prob\s+(\d+)/(\d+)")
CELL_RE = re.compile(r"\s*\[([^\]]+)\]\s+(\w+)\s+seed=(\d+)")
RESUME_MARKER = "--- autopilot resume "


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    line = f"[watchdog {datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _load_policy() -> Dict[str, Any]:
    if POLICY_PATH.is_file():
        try:
            return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"policy": {"restart_eval_on_stall_min": 45, "watchdog_poll_min": 10}}


def _load_status() -> Dict[str, Any]:
    if STATUS_PATH.is_file():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _pid_alive_win(pid: int) -> bool:
    if pid <= 0:
        return False
    proc = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout or ""
    return str(pid) in out and "No tasks" not in out


def _find_process(substr: str) -> Optional[int]:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Where-Object { $_.CommandLine -match '" + substr.replace("'", "''") + "' } "
        "| Select-Object -First 1 -ExpandProperty ProcessId"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        out = (proc.stdout or "").strip()
        return int(out) if out.isdigit() else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def _active_run_lines(text: str) -> list[str]:
    """Only parse the latest eval run (after last autopilot resume marker)."""
    if RESUME_MARKER in text:
        text = text.rsplit(RESUME_MARKER, 1)[-1]
    return text.splitlines()


def _wave2_progress() -> Tuple[int, int, float]:
    """Return (prob_cur, prob_total, log_stale_sec) for the active eval run."""
    if not WAVE2_LOG.is_file():
        return 0, 0, 1e9
    stale = max(0.0, time.time() - WAVE2_LOG.stat().st_mtime)
    text = WAVE2_LOG.read_text(encoding="utf-8", errors="replace")[-200_000:]
    lines = _active_run_lines(text)
    cur, total = 0, 0
    cell_start = 0
    for i, line in enumerate(lines):
        if CELL_RE.search(line):
            cell_start = i + 1
            cur, total = 0, 0
    for line in lines[cell_start:]:
        m = PROB_RE.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
    return cur, total, stale


def _wave2_incomplete(st: Dict[str, Any]) -> bool:
    if _wave2_complete(st):
        return False
    final_json = ROOT / "results" / "headline_w2_primary_full" / "table2_results.json"
    return not final_json.is_file()


def _kill_process(substr: str, reason: str) -> None:
    pid = _find_process(substr)
    if not pid or not _pid_alive_win(pid):
        return
    _log(f"kill {substr} pid={pid} ({reason})")
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/F", "/T"],
        capture_output=True,
        text=True,
        check=False,
    )


def _kill_wave2_stack(reason: str) -> None:
    _kill_process("run_cts_eval_full", reason)
    _kill_process("post_s2_autopilot", reason)


def _pipeline_complete(st: Dict[str, Any]) -> bool:
    if st.get("final_verdict") == "PASS":
        return True
    done = set(st.get("completed_step_ids") or [])
    return "15_rebuttal" in done


def _wave2_complete(st: Dict[str, Any]) -> bool:
    w2 = st.get("steps", {}).get("08_wave2", {})
    if w2.get("status") == "PASS":
        final_json = ROOT / "results" / "headline_w2_primary_full" / "table2_results.json"
        return final_json.is_file()
    return False


def _should_restart(st: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[bool, str]:
    if _pipeline_complete(st):
        return False, "pipeline_complete"

    stall_min = float(policy.get("policy", {}).get("restart_eval_on_stall_min", 45))
    ap_pid = _find_process("post_s2_autopilot")
    ev_pid = _find_process("run_cts_eval_full")
    ap_alive = bool(ap_pid and _pid_alive_win(ap_pid))
    ev_alive = bool(ev_pid and _pid_alive_win(ev_pid))
    prob_cur, prob_total, stale = _wave2_progress()
    stale_min = stale / 60.0

    if _wave2_complete(st):
        if ap_alive:
            return False, f"autopilot_alive pid={ap_pid}"
        return True, "autopilot_dead_post_wave2"

    if _wave2_incomplete(st):
        # Primary signal: log freshness — even if autopilot PID is still alive.
        if stale < stall_min * 60:
            if ev_alive:
                return False, (
                    f"eval_alive pid={ev_pid} prob={prob_cur}/{prob_total} "
                    f"log_stale={stale:.0f}s"
                )
            if ap_alive:
                return False, (
                    f"wave2_active ap={ap_pid} prob={prob_cur}/{prob_total} "
                    f"log_stale={stale:.0f}s"
                )
            return False, f"wave2_recent log_stale={stale:.0f}s prob={prob_cur}/{prob_total}"

        return True, (
            f"wave2_stalled stale={stale_min:.0f}min ap={ap_alive} ev={ev_alive} "
            f"prob={prob_cur}/{prob_total}"
        )

    if ap_alive:
        return False, f"autopilot_alive pid={ap_pid}"
    if ev_alive:
        return False, f"eval_alive pid={ev_pid}"
    return True, "autopilot_and_eval_dead"


def _restart_autopilot(reason: str) -> bool:
    _log(f"RESTART autopilot: {reason}")
    _kill_wave2_stack(reason)
    time.sleep(2.0)
    st = _load_status()
    st.setdefault("watchdog_restarts", []).append(
        {"at_utc": _utc(), "reason": reason}
    )
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")

    try:
        proc = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER),
                "-ForceRestart",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        _log(f"launcher rc={proc.returncode} {out.strip()[:200]}")
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        _log("launcher timeout")
        return False


def run_once() -> int:
    policy = _load_policy()
    st = _load_status()
    restart, reason = _should_restart(st, policy)
    if restart:
        ok = _restart_autopilot(reason)
        return 0 if ok else 1
    _log(f"OK: {reason}")
    return 0


def run_watch(poll_min: float) -> int:
    _log(f"watchdog started poll={poll_min}min policy={POLICY_PATH.name}")
    try:
        while True:
            run_once()
            time.sleep(max(60.0, poll_min * 60.0))
    except KeyboardInterrupt:
        _log("watchdog stopped")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Autopilot watchdog")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-min", type=float, default=None)
    args = ap.parse_args()

    policy = _load_policy()
    poll = args.poll_min
    if poll is None:
        poll = float(policy.get("policy", {}).get("watchdog_poll_min", 10))

    if args.watch:
        return run_watch(poll)
    return run_once()


if __name__ == "__main__":
    raise SystemExit(main())
