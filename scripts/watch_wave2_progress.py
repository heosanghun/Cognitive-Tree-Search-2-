#!/usr/bin/env python3
"""Live Wave 2 / headline eval progress with visual loading bars.

Lightweight: no subprocess per refresh (avoids 'can't start new thread' on Windows
while GPU eval is running). Process checks are cached and use ctypes only.

Usage:
  python scripts/watch_wave2_progress.py --watch
  python scripts/watch_wave2_progress.py --watch --interval 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOG = ROOT / "results" / "post_s2_autopilot" / "logs" / "wave2.log"
DEFAULT_OUT = ROOT / "results" / "headline_w2_primary_full"
STATUS_PATH = ROOT / "results" / "post_s2_autopilot" / "autopilot_status.json"
AUTOPILOT_LOG = ROOT / "results" / "post_s2_autopilot" / "logs" / "autopilot.log"
PID_CACHE_PATH = ROOT / "results" / "post_s2_autopilot" / "watch_pids.json"

WAVE2_METHODS = ["cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop"]
WAVE2_BENCHES = ["math500", "gsm8k", "aime"]
WAVE2_SEEDS = [0, 1, 2, 3, 4]
BENCH_SIZES = {"math500": 500, "gsm8k": 1319, "aime": 30}

CELL_RE = re.compile(r"\s*\[([^\]]+)\]\s+(\w+)\s+seed=(\d+)")
PROB_RE = re.compile(r"prob\s+(\d+)/(\d+)\s+.*?time=([\d.]+)s")
PROB_SIMPLE_RE = re.compile(r"prob\s+(\d+)/(\d+)")
FALLBACK_RE = re.compile(r"prob\s+(\d+)/(\d+)\s+fallback\s+time=([\d.]+)s")

SPINNER = "|/-\\"
GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
CYN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RST = "\033[0m"

# Windows PROCESS_QUERY_LIMITED_INFORMATION
_WIN_QUERY = 0x1000


@dataclass
class Snapshot:
    method: Optional[str] = None
    bench: Optional[str] = None
    seed: Optional[int] = None
    prob_cur: int = 0
    prob_total: int = 0
    last_time_s: float = 0.0
    log_mtime: float = 0.0
    log_bytes: int = 0
    eval_pid: Optional[int] = None
    eval_alive: bool = False
    autopilot_alive: bool = False
    autopilot_pid: Optional[int] = None
    autopilot_step: str = "unknown"
    partial_cells_done: int = 0
    health: str = "UNKNOWN"
    stale_sec: float = 0.0


@dataclass
class WatchState:
    last_prob_cur: int = -1
    last_change_ts: float = field(default_factory=time.time)
    tick: int = 0
    eval_pid: Optional[int] = None
    autopilot_pid: Optional[int] = None
    pid_scan_ts: float = 0.0


def _enable_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        h = ctypes.windll.kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(mode))
        ctypes.windll.kernel32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


def _clear_screen() -> None:
    # No subprocess — avoids extra threads while GPU eval runs.
    print("\033[2J\033[H", end="")


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(_WIN_QUERY, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_pid_cache() -> Tuple[Optional[int], Optional[int]]:
    if not PID_CACHE_PATH.is_file():
        return None, None
    try:
        data = json.loads(PID_CACHE_PATH.read_text(encoding="utf-8"))
        ev = data.get("eval_pid")
        ap = data.get("autopilot_pid")
        return (int(ev) if ev else None), (int(ap) if ap else None)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None


def _save_pid_cache(eval_pid: Optional[int], autopilot_pid: Optional[int]) -> None:
    PID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_CACHE_PATH.write_text(
        json.dumps(
            {
                "eval_pid": eval_pid,
                "autopilot_pid": autopilot_pid,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _scan_pids_once() -> Tuple[Optional[int], Optional[int]]:
    """One-shot scan via ctypes Toolhelp32 — no PowerShell/subprocess."""
    eval_pid: Optional[int] = None
    ap_pid: Optional[int] = None
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        TH32CS_SNAPPROCESS = 0x00000002
        kernel32 = ctypes.windll.kernel32

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == ctypes.c_void_p(-1).value:
            return _load_pid_cache()

        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)

        # Full cmdline needs QueryFullProcessImageName / WMI; match python.exe PIDs
        # then verify with OpenProcess + read autopilot log for correlation.
        python_pids: List[int] = []
        if kernel32.Process32First(snap, ctypes.byref(entry)):
            while True:
                name = entry.szExeFile.decode("utf-8", errors="replace").lower()
                if name == "python.exe":
                    python_pids.append(int(entry.th32ProcessID))
                if not kernel32.Process32Next(snap, ctypes.byref(entry)):
                    break
        kernel32.CloseHandle(snap)

        # Prefer cached PIDs if still alive among python processes.
        cached_ev, cached_ap = _load_pid_cache()
        if cached_ev in python_pids and _pid_alive(cached_ev):
            eval_pid = cached_ev
        if cached_ap in python_pids and _pid_alive(cached_ap):
            ap_pid = cached_ap

        # Fallback: read autopilot status file pid hint.
        if STATUS_PATH.is_file():
            try:
                st = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
                hint = st.get("autopilot_pid")
                if hint and int(hint) in python_pids and _pid_alive(int(hint)):
                    ap_pid = int(hint)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if eval_pid or ap_pid:
            _save_pid_cache(eval_pid, ap_pid)
        return eval_pid, ap_pid
    except Exception:
        return _load_pid_cache()


def _resolve_pids(ws: WatchState, *, force_scan: bool = False) -> Tuple[Optional[int], bool, Optional[int], bool]:
    now = time.time()
    if force_scan or (now - ws.pid_scan_ts) > 60.0:
        ev, ap = _scan_pids_once()
        if ev:
            ws.eval_pid = ev
        if ap:
            ws.autopilot_pid = ap
        ws.pid_scan_ts = now

    if ws.eval_pid is None or ws.autopilot_pid is None:
        cev, cap = _load_pid_cache()
        ws.eval_pid = ws.eval_pid or cev
        ws.autopilot_pid = ws.autopilot_pid or cap

    eval_alive = _pid_alive(ws.eval_pid)
    ap_alive = _pid_alive(ws.autopilot_pid)
    return ws.eval_pid, eval_alive, ws.autopilot_pid, ap_alive


def _tail_text(path: Path, max_bytes: int = 512_000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        return fh.read().decode("utf-8", errors="replace")


def _cell_plan() -> List[Tuple[str, int, str]]:
    plan: List[Tuple[str, int, str]] = []
    for method in WAVE2_METHODS:
        for seed in WAVE2_SEEDS:
            for bench in WAVE2_BENCHES:
                plan.append((method, seed, bench))
    return plan


def _total_problems() -> int:
    return sum(BENCH_SIZES[bench] for _, _, bench in _cell_plan())


def _cell_index(method: str, seed: int, bench: str) -> Optional[int]:
    for i, item in enumerate(_cell_plan()):
        if item == (method, seed, bench):
            return i
    return None


def _problems_before_cell(cell_idx: int) -> int:
    total = 0
    for i in range(cell_idx):
        total += BENCH_SIZES.get(_cell_plan()[i][2], 0)
    return total


def _read_autopilot_step() -> str:
    if STATUS_PATH.is_file():
        try:
            st = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if "08_wave2" not in st.get("completed_step_ids", []):
                if "07_compare_w1" in st.get("completed_step_ids", []):
                    return "08_wave2"
            for sid in ("15_rebuttal", "12_zip", "09_docs", "08_wave2"):
                if sid in st.get("steps", {}) and sid not in st.get("completed_step_ids", []):
                    return sid
        except json.JSONDecodeError:
            pass
    text = _tail_text(AUTOPILOT_LOG, 32_000)
    for line in reversed(text.splitlines()):
        if "running 08_wave2" in line:
            return "08_wave2"
        if "running 09_docs" in line:
            return "09_docs"
        if "autopilot complete" in line:
            return "DONE"
    return "unknown"


def _partial_cells_done(out_dir: Path) -> int:
    partial = out_dir / "table2_results.partial.json"
    if not partial.is_file():
        return 0
    try:
        data = json.loads(partial.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    raw = data.get("raw_scores") or {}
    return sum(
        len((raw.get(method) or {}).get(bench) or [])
        for method in WAVE2_METHODS
        for bench in WAVE2_BENCHES
    )


def _active_run_lines(text: str) -> List[str]:
    """Keep only lines from the latest eval run (after last autopilot resume marker)."""
    marker = "--- autopilot resume "
    if marker in text:
        text = text.rsplit(marker, 1)[-1]
    return text.splitlines()


def parse_log(text: str) -> Snapshot:
    snap = Snapshot()
    lines = _active_run_lines(text)
    current_method: Optional[str] = None
    current_bench: Optional[str] = None
    current_seed: Optional[int] = None
    last_prob_line = ""
    cell_start = 0

    for i, line in enumerate(lines):
        cm = CELL_RE.search(line)
        if cm:
            current_method, current_bench, current_seed = cm.group(1), cm.group(2), int(cm.group(3))
            cell_start = i + 1
            last_prob_line = ""

    snap.method = current_method
    snap.bench = current_bench
    snap.seed = current_seed

    for line in lines[cell_start:]:
        if "prob " in line and "/" in line:
            last_prob_line = line

    if last_prob_line:
        m = PROB_RE.search(last_prob_line) or FALLBACK_RE.search(last_prob_line)
        if m:
            snap.prob_cur = int(m.group(1))
            snap.prob_total = int(m.group(2))
            snap.last_time_s = float(m.group(3))
        else:
            m2 = PROB_SIMPLE_RE.search(last_prob_line)
            if m2:
                snap.prob_cur = int(m2.group(1))
                snap.prob_total = int(m2.group(2))

    snap.autopilot_step = _read_autopilot_step()
    return snap


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "n/a"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d}d {h}h"


def _block_bar(pct: float, width: int = 50, fill: str = "#", empty: str = ".") -> str:
    pct = max(0.0, min(100.0, pct))
    n = int(round(width * pct / 100.0))
    return fill * n + empty * (width - n)


def _unicode_bar(pct: float, width: int = 40) -> str:
    """Terminal progress bar (ASCII on Windows cp949, blocks on UTF-8)."""
    pct = max(0.0, min(100.0, pct))
    n = int(round(width * pct / 100.0))
    if sys.platform == "win32":
        enc = getattr(sys.stdout, "encoding", None) or ""
        if "utf" not in enc.lower():
            return "#" * n + "." * (width - n)
    return "\u2588" * n + "\u2591" * (width - n)


def _recent_avg_time(text: str, n: int = 20) -> float:
    times: List[float] = []
    for line in text.splitlines():
        m = PROB_RE.search(line) or FALLBACK_RE.search(line)
        if m:
            times.append(float(m.group(3)))
    if not times:
        return 0.0
    sample = times[-n:]
    return sum(sample) / len(sample)


def _health(snap: Snapshot, ws: WatchState) -> str:
    # Primary signal: log freshness (no subprocess needed).
    if snap.prob_cur > 0 and snap.stale_sec < 360:
        return "RUNNING"
    if snap.eval_alive and snap.stale_sec < 600:
        return "RUNNING"
    if snap.eval_alive and snap.stale_sec < 1800:
        return "SLOW"
    if snap.stale_sec < 600 and snap.prob_cur != ws.last_prob_cur:
        return "RUNNING"
    if snap.stale_sec < 3600:
        return "STALLED"
    return "STOPPED"


def _health_color(health: str) -> str:
    return {
        "RUNNING": GRN,
        "SLOW": YLW,
        "STALLED": YLW,
        "STOPPED": RED,
        "UNKNOWN": DIM,
    }.get(health, DIM)


def _inflight_pct(snap: Snapshot, avg_t: float) -> float:
    if avg_t <= 0:
        return 0.0
    if snap.health not in ("RUNNING", "SLOW"):
        return 0.0
    elapsed = min(snap.stale_sec, avg_t * 1.5)
    return 100.0 * elapsed / avg_t


def render(
    snap: Snapshot,
    log_path: Path,
    out_dir: Path,
    log_text: str,
    ws: WatchState,
    *,
    use_color: bool = True,
) -> str:
    def c(code: str, text: str) -> str:
        return f"{code}{text}{RST}" if use_color else text

    lines: List[str] = []
    now = datetime.now()
    spin = SPINNER[ws.tick % len(SPINNER)]
    health = snap.health
    hc = _health_color(health)

    lines.append("")
    lines.append(c(BOLD, "=" * 70))
    lines.append(c(BOLD, f"  CTS Wave 2  {spin}  LIVE PROGRESS"))
    lines.append(c(BOLD, "=" * 70))
    lines.append(f"  Time     : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"  Status   : {c(hc, health)}"
        f"  |  log idle {snap.stale_sec:.0f}s"
        f"  |  step {snap.autopilot_step}"
    )
    pid_note = ""
    if snap.eval_pid:
        pid_note = f"eval PID {snap.eval_pid} ({'alive' if snap.eval_alive else 'dead'})"
    else:
        pid_note = "eval PID (log-based; use --scan-pids once if needed)"
    lines.append(f"  Process  : {pid_note}")

    if health == "STOPPED":
        lines.append("")
        lines.append(c(RED, "  !! Eval appears STOPPED (log idle > 1h)."))
        lines.append(c(RED, "  !! Restart: powershell -File scripts\\start_post_s2_autopilot.ps1"))
    elif health == "SLOW":
        lines.append("")
        lines.append(c(YLW, "  ~ Long problem in flight (~3 min each is normal for cts_4nu)."))

    total_problems = _total_problems()
    total_cells = len(_cell_plan())
    partial_done = snap.partial_cells_done

    if snap.method and snap.bench is not None and snap.seed is not None and snap.prob_total > 0:
        cell_idx = _cell_index(snap.method, snap.seed, snap.bench)
        cell_no = (cell_idx + 1) if cell_idx is not None else "?"
        cell_pct = 100.0 * snap.prob_cur / snap.prob_total
        avg_t = _recent_avg_time(log_text)
        inflight = _inflight_pct(snap, avg_t)
        display_pct = min(99.9, cell_pct + inflight / snap.prob_total)

        delta = ""
        if ws.last_prob_cur >= 0 and snap.prob_cur > ws.last_prob_cur:
            delta = c(GRN, f"  (+{snap.prob_cur - ws.last_prob_cur} since last refresh)")

        lines.append("")
        lines.append(c(CYN, f"  CURRENT CELL  ({cell_no}/{total_cells})"))
        lines.append(f"  [{snap.method}]  {snap.bench}  seed={snap.seed}")
        lines.append("")
        lines.append(c(BOLD, f"  >> CELL PROGRESS   {cell_pct:6.2f}%   ({snap.prob_cur} / {snap.prob_total})"))
        if cell_idx is not None:
            cell_overall_pct = 100.0 * (cell_no - 1) / total_cells
            lines.append(
                c(DIM, f"     (cells done: {cell_no - 1}/{total_cells} = "
                f"{cell_overall_pct:.2f}%)")
            )
        ubar = _unicode_bar(display_pct, 40)
        lines.append(f"  CELL  [{c(GRN if health == 'RUNNING' else YLW, ubar)}] {display_pct:5.1f}%{delta}")
        lines.append(
            f"  Problem  {snap.prob_cur:4d} / {snap.prob_total}   "
            f"{display_pct:5.1f}%"
        )
        bar_core = _block_bar(display_pct, 50, "#", ".")
        if health in ("RUNNING", "SLOW") and inflight > 0 and snap.prob_cur < snap.prob_total:
            tail = _block_bar(min(100.0, inflight / snap.prob_total * 100), 6, ">", ".")
            bar_core = _block_bar(cell_pct, 44, "#", ".") + tail
        lines.append(f"  |{c(GRN if health == 'RUNNING' else YLW, bar_core)}|")

        if snap.last_time_s > 0:
            lines.append(f"  Last done: {snap.last_time_s:.0f}s/problem  |  avg(last 20): {avg_t:.0f}s")

        if cell_idx is not None:
            done_global = _problems_before_cell(cell_idx) + snap.prob_cur
            global_pct = 100.0 * done_global / total_problems
            cell_left = max(0, snap.prob_total - snap.prob_cur)
            cell_eta = cell_left * avg_t if avg_t > 0 else 0

            lines.append("")
            lines.append(c(CYN, "  WAVE 2 TOTAL"))
            lines.append(c(BOLD, f"  >> TOTAL PROGRESS  {global_pct:6.2f}%   ({done_global} / {total_problems})"))
            lines.append(f"  {done_global:6d} / {total_problems} problems")
            tbar = _unicode_bar(global_pct, 40)
            lines.append(f"  TOTAL [{c(CYN, tbar)}] {global_pct:5.2f}%")
            lines.append(f"  |{_block_bar(global_pct, 50, '=', '.')}|")
            if partial_done > 0:
                cells_pct = 100.0 * partial_done / total_cells
                cbar = _unicode_bar(cells_pct, 40)
                lines.append(f"  CELLS [{c(CYN, cbar)}] {partial_done}/{total_cells} ({cells_pct:.1f}%)")
            if avg_t > 0:
                lines.append(f"  Cell ETA : {_fmt_duration(cell_eta)}  ({cell_left} problems left in cell)")
    elif snap.method and snap.bench is not None and snap.seed is not None:
        lines.append("")
        lines.append(c(CYN, f"  CURRENT CELL  : [{snap.method}] {snap.bench} seed={snap.seed}"))
        lines.append(c(YLW, "  Model loading or first problem starting..."))
        lines.append(f"  |{_block_bar(min(100.0, snap.stale_sec / 180.0 * 100), 50, '>', '.')}|")
    else:
        lines.append("")
        lines.append("  Waiting for eval output...")

    lines.append("")
    lines.append(c(DIM, "  Ctrl+C to exit  |  refresh: python scripts/watch_wave2_progress.py --watch"))
    lines.append(c(BOLD, "=" * 70))
    return "\n".join(lines)


def _collect(
    log_path: Path,
    out_dir: Path,
    ws: WatchState,
    *,
    check_pids: bool,
) -> Tuple[Snapshot, str]:
    log_text = _tail_text(log_path)
    snap = parse_log(log_text)
    if log_path.is_file():
        st = log_path.stat()
        snap.log_mtime = st.st_mtime
        snap.log_bytes = st.st_size
        snap.stale_sec = max(0.0, time.time() - st.st_mtime)
    snap.partial_cells_done = _partial_cells_done(out_dir)

    if check_pids:
        ep, ea, ap, aa = _resolve_pids(ws)
        snap.eval_pid = ep
        snap.eval_alive = ea
        snap.autopilot_pid = ap
        snap.autopilot_alive = aa

    snap.health = _health(snap, ws)

    if snap.prob_cur != ws.last_prob_cur:
        ws.last_prob_cur = snap.prob_cur
        ws.last_change_ts = time.time()
    return snap, log_text


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Wave 2 eval progress (visual bars)")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument(
        "--scan-pids",
        action="store_true",
        help="scan process list once (optional; watch mode uses log-only by default)",
    )
    args = ap.parse_args()

    _enable_ansi()
    use_color = not args.no_color
    ws = WatchState()

    if args.scan_pids or not args.watch:
        ev, ap = _scan_pids_once()
        ws.eval_pid = ev
        ws.autopilot_pid = ap
        ws.pid_scan_ts = time.time()

    if not args.watch:
        snap, log_text = _collect(args.log, args.output_dir, ws, check_pids=True)
        print(render(snap, args.log, args.output_dir, log_text, ws, use_color=use_color))
        return 0

    try:
        while True:
            ws.tick += 1
            # Watch loop: log-only refresh; PID scan at most every 60s if cached.
            snap, log_text = _collect(
                args.log,
                args.output_dir,
                ws,
                check_pids=(ws.tick == 1 or (time.time() - ws.pid_scan_ts) > 60),
            )
            _clear_screen()
            print(render(snap, args.log, args.output_dir, log_text, ws, use_color=use_color), flush=True)
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
        return 0
    except Exception as exc:
        print(f"\n[watch] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
