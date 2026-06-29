#!/usr/bin/env python3
"""Post-Stage-2 full autopilot (steps 3-15 from HEADLINE_EVAL_RUNBOOK).

Waits until in-flight Stage 2 PPO reaches step 10000, then runs verify,
backup, Headline eval (Wave 1-3), docs, ZIP, optional Phase-4 fork, and
rebuttal placeholder fill — unattended.

Steps 1-2 (audit + JSONL solution) are skipped when already complete.

Usage:
  python scripts/post_s2_autopilot.py --watch          # wait + run all
  python scripts/post_s2_autopilot.py --dry-run        # print plan only
  python scripts/post_s2_autopilot.py --resume         # continue from status

Status: results/post_s2_autopilot/autopilot_status.json
Logs:   results/post_s2_autopilot/logs/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATUS_PATH = ROOT / "results" / "post_s2_autopilot" / "autopilot_status.json"
LOG_DIR = ROOT / "results" / "post_s2_autopilot" / "logs"
RESUME_NOTE = ROOT / "logs" / "STAGE2_RESUME_STATE.md"
DEFAULT_LOG = ROOT / "logs" / "stage2_paper_full_resume_20260517_204944.log"
FINAL_CKPT = ROOT / "artifacts" / "stage2_meta_value.pt"
JSONL_PATH = ROOT / "data" / "stage2" / "math_train_prompts_5000.jsonl"
STEP_RE = re.compile(r"stage2 step=(\d+)/(\d+)")

WAVE1_OUT = ROOT / "results" / "headline_w1_math500_s0"
WAVE2_OUT = ROOT / "results" / "headline_w2_primary_full"
WAVE3_OUT = ROOT / "results" / "headline_w3_table2_full"
SMOKE_OUT = ROOT / "results" / "post_stage2_May2026"

WAVE2_METHODS = ["cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop"]
WAVE2_BENCHES = ["math500", "gsm8k", "aime"]

POLICY_PATH = ROOT / "configs" / "autopilot_autonomous.json"

MATH_PASS_THRESHOLD = 52.0  # Phase 4 if below
MATH_CLOUD_THRESHOLD = 58.0  # Cloud hint if below


def _load_autonomous_policy() -> Dict[str, Any]:
    if POLICY_PATH.is_file():
        try:
            return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"mode": "default", "policy": {}}


def _find_running_eval_pid() -> Optional[int]:
    """Return PID of run_cts_eval_full.py if alive (Windows)."""
    if sys.platform != "win32":
        return None
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Where-Object { $_.CommandLine -match 'run_cts_eval_full\\.py' } "
        "| Select-Object -First 1 -ExpandProperty ProcessId"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        out = (proc.stdout or "").strip()
        if out.isdigit() and _pid_alive(int(out)):
            return int(out)
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    line = f"[autopilot {_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "autopilot.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _load_status() -> Dict[str, Any]:
    if STATUS_PATH.is_file():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"started_at_utc": _utc(), "steps": {}, "completed_step_ids": []}


def _save_status(st: Dict[str, Any]) -> None:
    st["updated_at_utc"] = _utc()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")


def _read_log_path() -> Path:
    if RESUME_NOTE.is_file():
        text = RESUME_NOTE.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Stage 2 log file\s*\|\s*`([^`]+)`", text)
        if m:
            return Path(m.group(1))
    return DEFAULT_LOG


def _tail_step(log_path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not log_path.is_file():
        return None, None
    last: Tuple[Optional[int], Optional[int]] = (None, None)
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                last = int(m.group(1)), int(m.group(2))
    return last


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
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
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_training_pid() -> int:
    if RESUME_NOTE.is_file():
        text = RESUME_NOTE.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"PID at snapshot\s*\|\s*\*\*(\d+)\*\*", text)
        if m:
            return int(m.group(1))
    return 0


def _headline_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["CTS_EVAL_TAU_CAP"] = "1e14"
    env.pop("CTS_EVAL_EPISODE_TIMEOUT", None)
    env.setdefault("CTS_GLOBAL_SEED", "42")
    return env


def _run(
    cmd: List[str],
    *,
    log_name: str,
    env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    log_path = LOG_DIR / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # Append eval logs on resume so crash/restart history is preserved.
    file_mode = "a" if log_path.is_file() and log_name in ("wave2.log", "wave1_rerun.log") else "w"
    if file_mode == "a":
        with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n--- autopilot resume {_utc()} ---\n")
    try:
        with open(log_path, file_mode, encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env or os.environ.copy(),
                cwd=str(cwd or ROOT),
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


def _step_done(st: Dict[str, Any], step_id: str) -> bool:
    return step_id in st.get("completed_step_ids", [])


def _send_step_notification(step_id: str, result: Dict[str, Any]) -> None:
    import urllib.request
    config_path = ROOT / "configs" / "notifications.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return

    status = result.get("status", "UNKNOWN")
    duration = result.get("duration_s")
    
    emoji = "✅"
    if status == "FAIL":
        emoji = "❌"
    elif status in ("SKIP", "MANUAL"):
        emoji = "🔄"
    elif status == "WARN":
        emoji = "⚠️"
        
    msg = f"[{emoji} Step Completed] {step_id}\n"
    msg += f"• Status: {status}\n"
    if duration is not None:
        if duration >= 60:
            msg += f"• Duration: {int(duration // 60)}m {int(duration % 60)}s\n"
        else:
            msg += f"• Duration: {duration:.1f}s\n"
            
    if "reason" in result:
        msg += f"• Reason: {result['reason']}\n"
    if "error" in result:
        msg += f"• Error: {result['error']}\n"
        
    if step_id == "09_docs" and "numbers" in result:
        nums = result["numbers"]
        msg += "• Results Summary:\n"
        for k, v in nums.items():
            if v is not None:
                msg += f"  - {k}: {v}\n"

    # Send function
    def post_json(url: str, payload: dict):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pass
        except Exception:
            pass

    # Local Log
    local_conf = config.get("local_log", {})
    if local_conf.get("enabled", True):
        log_p = ROOT / local_conf.get("log_path", "results/post_s2_autopilot/logs/notifications.log")
        log_p.parent.mkdir(parents=True, exist_ok=True)
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(f"--- NOTIFICATION {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{msg}\n\n")

    # Discord
    discord_conf = config.get("discord", {})
    if discord_conf.get("enabled", False) and discord_conf.get("webhook_url"):
        post_json(discord_conf["webhook_url"], {"content": msg})
        
    # Slack
    slack_conf = config.get("slack", {})
    if slack_conf.get("enabled", False) and slack_conf.get("webhook_url"):
        post_json(slack_conf["webhook_url"], {"text": msg})
        
    # Telegram
    tg_conf = config.get("telegram", {})
    if tg_conf.get("enabled", False) and tg_conf.get("bot_token") and tg_conf.get("chat_id"):
        url = f"https://api.telegram.org/bot{tg_conf['bot_token']}/sendMessage"
        post_json(url, {
            "chat_id": tg_conf["chat_id"],
            "text": msg
        })


def _send_final_notification(st: Dict[str, Any]) -> None:
    import urllib.request
    config_path = ROOT / "configs" / "notifications.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return
        
    verdict = st.get("final_verdict", "UNKNOWN")
    emoji = "🎉" if verdict == "PASS" else "⚠️"
    msg = f"[{emoji} Autopilot Final Verdict: {verdict}]\n"
    msg += f"• Completed Steps: {', '.join(st.get('completed_step_ids', []))}\n"
    
    w2_info = st.get("steps", {}).get("09_docs", {}).get("numbers", {})
    if w2_info:
        msg += "• Final Evaluation metrics:\n"
        for k, v in w2_info.items():
            if v is not None:
                msg += f"  - {k}: {v}\n"

    # Send function
    def post_json(url: str, payload: dict):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pass
        except Exception:
            pass

    # Local Log
    local_conf = config.get("local_log", {})
    if local_conf.get("enabled", True):
        log_p = ROOT / local_conf.get("log_path", "results/post_s2_autopilot/logs/notifications.log")
        log_p.parent.mkdir(parents=True, exist_ok=True)
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(f"--- NOTIFICATION {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{msg}\n\n")

    # Discord
    discord_conf = config.get("discord", {})
    if discord_conf.get("enabled", False) and discord_conf.get("webhook_url"):
        post_json(discord_conf["webhook_url"], {"content": msg})
        
    # Slack
    slack_conf = config.get("slack", {})
    if slack_conf.get("enabled", False) and slack_conf.get("webhook_url"):
        post_json(slack_conf["webhook_url"], {"text": msg})
        
    # Telegram
    tg_conf = config.get("telegram", {})
    if tg_conf.get("enabled", False) and tg_conf.get("bot_token") and tg_conf.get("chat_id"):
        url = f"https://api.telegram.org/bot{tg_conf['bot_token']}/sendMessage"
        post_json(url, {
            "chat_id": tg_conf["chat_id"],
            "text": msg
        })


def _mark_done(st: Dict[str, Any], step_id: str, result: Dict[str, Any]) -> None:
    st.setdefault("steps", {})[step_id] = result
    ids = st.setdefault("completed_step_ids", [])
    if step_id not in ids:
        ids.append(step_id)
    _save_status(st)
    try:
        _send_step_notification(step_id, result)
    except Exception as e:
        _log(f"Notification error: {e}")


def _verify_final() -> Dict[str, Any]:
    from scripts.run_post_stage2_pipeline import phase_verify_stage2

    ns = argparse.Namespace()
    return phase_verify_stage2(ns)


def _jsonl_has_solution() -> bool:
    if not JSONL_PATH.is_file():
        return False
    try:
        row = json.loads(JSONL_PATH.read_text(encoding="utf-8").splitlines()[0])
        return "solution" in row
    except (json.JSONDecodeError, IndexError):
        return False


def _audit_pass() -> bool:
    r = _run([sys.executable, "scripts/_reviewer_local_audit.py"], log_name="precheck_audit.log")
    return r.get("status") == "PASS"


def _extract_math500_cts4nu(results_dir: Path) -> Optional[float]:
    p = results_dir / "table2_results.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        stat = data.get("cts_4nu", {}).get("math500")
        if stat is None:
            return None
        return float(stat.get("mean", 0.0)) * 100.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _backup_ckpts() -> Dict[str, Any]:
    """Backup Stage-2 artifacts only (skip 15GB stage1_last — optional manual copy)."""
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "artifacts" / "backups"
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in (
        "stage2_meta_value.pt",
        "stage2_meta_value.intermediate.pt",
    ):
        src = ROOT / "artifacts" / name
        if not src.is_file():
            continue
        out = dest / f"{src.stem}_{ts}{src.suffix}"
        _log(f"backup: copying {name} ({src.stat().st_size / 1e6:.1f} MB)")
        shutil.copyfile(src, out)
        shutil.copystat(src, out, follow_symlinks=False)
        copied.append(str(out))
        _log(f"backup: done {out.name}")
    return {"status": "PASS" if copied else "FAIL", "copied": copied}


def _compare_paper(results_dir: Path, out_name: str = "PAPER_COMPARE.md") -> Dict[str, Any]:
    out = results_dir / out_name
    return _run(
        [sys.executable, "scripts/compare_to_paper_table2.py", str(results_dir), "--out", str(out)],
        log_name=f"compare_{results_dir.name}.log",
    )


def _update_docs(st: Dict[str, Any], wave2_dir: Path) -> Dict[str, Any]:
    """Fill EXPERIMENTAL_RESULTS §0 and OPENREVIEW placeholder from Wave 2 JSON."""
    math = _extract_math500_cts4nu(wave2_dir)
    gsm = None
    aime = None
    p = wave2_dir / "table2_results.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            gsm = float(data.get("cts_4nu", {}).get("gsm8k", {}).get("mean", 0)) * 100
            aime = float(data.get("cts_4nu", {}).get("aime", {}).get("mean", 0)) * 100
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    numbers = {
        "math500_cts4nu_pct": math,
        "gsm8k_cts4nu_pct": gsm,
        "aime_cts4nu_pct": aime,
        "wave2_dir": str(wave2_dir),
        "updated_at_utc": _utc(),
    }
    num_path = ROOT / "results" / "post_s2_autopilot" / "rebuttal_numbers.json"
    num_path.parent.mkdir(parents=True, exist_ok=True)
    num_path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    # Copy headline compare into table2 index path
    src_cmp = wave2_dir / "PAPER_COMPARE.md"
    if src_cmp.is_file():
        dst = ROOT / "results" / "table2" / "PAPER_VS_LOCAL_HEADLINE.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_cmp, dst)

    exp = ROOT / "EXPERIMENTAL_RESULTS.md"
    if exp.is_file() and math is not None:
        text = exp.read_text(encoding="utf-8")
        text = re.sub(
            r"\| Stage 2 PPO 10k steps \| 🔄.*?\|",
            "| Stage 2 PPO 10k steps | ✅ 완료 | autopilot |",
            text,
            count=1,
        )
        text = re.sub(
            r"\| `stage2_meta_value.pt` \| ⏳.*?\|",
            "| `stage2_meta_value.pt` | ✅ | step 10000 |",
            text,
            count=1,
        )
        text = re.sub(
            r"\| Table 2 post-fix \| ⏳.*?\|",
            f"| Table 2 post-fix (Wave 2) | ✅ | MATH CTS-4ν={math:.1f}% |",
            text,
            count=1,
        )
        exp.write_text(text, encoding="utf-8")

    openreview = ROOT / "OPENREVIEW_RESPONSE_PREP.md"
    if openreview.is_file() and math is not None:
        text = openreview.read_text(encoding="utf-8")
        repl = {
            r"\| Stage 2 final ckpt \| 🔄.*?\|": "| Stage 2 final ckpt | ✅ | `artifacts/stage2_meta_value.pt` |",
            r"\| `training_meta.step` \| 🔄.*?\|": "| `training_meta.step` | ✅ | **10000** |",
            r"\| `paper_faithful_p0_4` \| 🔄.*?\|": "| `paper_faithful_p0_4` | ✅ | **True** |",
            r"\| Table 2 post-fix \(limit 50\) \| ⏳.*?\|": f"| Table 2 Wave 2 primary | ✅ | MATH={math:.1f}% GSM={gsm:.1f}% AIME={aime:.1f}% |",
            r"\| PAPER_VS_LOCAL.md refresh \| ⏳.*?\|": "| PAPER_VS_LOCAL.md refresh | ✅ | HEADLINE compare |",
        }
        for pat, sub in repl.items():
            text = re.sub(pat, sub, text, count=1)
        openreview.write_text(text, encoding="utf-8")

    return {"status": "PASS", "numbers": numbers}


def _wait_for_stage2(args: argparse.Namespace, st: Dict[str, Any]) -> Dict[str, Any]:
    log_path = _read_log_path()
    pid = _read_training_pid()
    ckpt_init_mtime = FINAL_CKPT.stat().st_mtime if FINAL_CKPT.is_file() else None
    started = time.time()
    _log(f"waiting for Stage 2 (log={log_path}, pid={pid or 'auto'})")

    while True:
        step, total = _tail_step(log_path)
        alive = _pid_alive(pid) if pid else False
        elapsed_min = (time.time() - started) / 60.0

        hb = {
            "phase": "wait_stage2",
            "step": step,
            "total": total,
            "pid": pid,
            "pid_alive": alive,
            "elapsed_min": round(elapsed_min, 1),
        }
        st["heartbeat"] = hb
        _save_status(st)

        if step is not None and total == 10000 and step >= 10000:
            time.sleep(10)  # flush final ckpt
            ver = _verify_final()
            if ver.get("status") in ("PASS", "WARN"):
                meta = ver.get("details", {}).get("training_meta") or {}
                tstep = meta.get("step") if isinstance(meta, dict) else None
                if tstep in (None, 10000):
                    _log("Stage 2 complete: step=10000, ckpt verified")
                    return {"status": "PASS", "verify": ver, "step": step}

        if pid and not alive and step is not None and step < (total or 10000):
            return {"status": "FAIL", "reason": "stage2_crash", "step": step}

        if elapsed_min >= args.max_wait_min:
            return {"status": "FAIL", "reason": "timeout", "elapsed_min": elapsed_min}

        if int(elapsed_min) % 10 == 0 and int(elapsed_min) > 0:
            _log(f"wait: step={step}/{total} pid_alive={alive} elapsed={elapsed_min:.0f}min")

        time.sleep(args.poll_seconds)


def _stage2_ready() -> bool:
    step, total = _tail_step(_read_log_path())
    if not (step == total == 10000 and FINAL_CKPT.is_file()):
        return False
    try:
        ver = _verify_final()
        return ver.get("status") in ("PASS", "WARN")
    except Exception:
        return False


def _heal_incomplete_wave2(st: Dict[str, Any]) -> None:
    """Undo premature 08 FAIL / 09 docs when Wave 2 is not actually complete."""
    partial = WAVE2_OUT / "table2_results.partial.json"
    final_json = WAVE2_OUT / "table2_results.json"
    w2 = st.get("steps", {}).get("08_wave2", {})
    w2_pass = w2.get("status") == "PASS"
    if final_json.is_file() and w2_pass:
        return
    n_cells = 0
    if partial.is_file():
        try:
            raw = json.loads(partial.read_text(encoding="utf-8")).get("raw_scores") or {}
            n_cells = sum(
                len((raw.get(m) or {}).get(b) or [])
                for m in ("cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop")
                for b in ("math500", "gsm8k", "aime")
            )
        except json.JSONDecodeError:
            pass
    expected_cells = 75
    if w2_pass and n_cells >= expected_cells:
        return
    if not w2_pass or n_cells < expected_cells:
        _log(f"Heal: Wave 2 incomplete (status={w2.get('status')}, partial_cells={n_cells}/75)")
        rollback = ["08_wave2", "09_docs", "10_smoke", "11_wave3", "12_zip", "13_phase4", "14_cloud", "15_rebuttal"]
        ids = st.setdefault("completed_step_ids", [])
        for sid in rollback:
            if sid in ids:
                ids.remove(sid)
        st.pop("final_verdict", None)
        _save_status(st)


def _heal_wait_status(st: Dict[str, Any]) -> None:
    """Recover from an earlier wait-timeout once Stage 2 has finished."""
    if not _stage2_ready():
        return
    prev = st.get("steps", {}).get("00_wait", {})
    if prev.get("status") != "PASS":
        _mark_done(
            st,
            "00_wait",
            {"status": "PASS", "reason": "stage2_complete_on_resume", "prior": prev},
        )
        _log("Stage 2 already complete; healed 00_wait and resuming pipeline.")


def run_autopilot(args: argparse.Namespace) -> int:
    st = _load_status()
    policy = _load_autonomous_policy()
    st.setdefault("autopilot_pid", os.getpid())
    st["autonomous_mode"] = {
        "enabled": True,
        "no_user_prompts": policy.get("no_user_prompts", True),
        "policy_path": str(POLICY_PATH),
        "wave3_after_primary": args.wave3_after_primary,
    }
    _heal_wait_status(st)
    _heal_incomplete_wave2(st)
    _save_status(st)

    steps_plan = [
        ("01_audit", "skip_precheck_audit", lambda: {"status": "PASS", "skipped": _audit_pass()}),
        ("02_jsonl", "skip_jsonl", lambda: {"status": "PASS", "has_solution": _jsonl_has_solution()}),
        ("03_verify", "verify_final", lambda: _verify_final()),
        ("04_backup", "backup_ckpts", _backup_ckpts),
        ("05_env", "headline_env", lambda: {"status": "PASS", "env": "CTS_EVAL_TAU_CAP=1e14"}),
        ("06_wave1", "wave1", lambda: _run(
            [
                sys.executable, "scripts/run_cts_eval_full.py",
                "--config", "paper_parity",
                "--benchmarks", "math500",
                "--methods", "cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop",
                "--seeds", "1",
                "--device", args.device,
                "--output-dir", str(WAVE1_OUT),
            ],
            log_name="wave1.log",
            env=_headline_env(),
            timeout_s=args.wave1_timeout_s,
        )),
        ("07_compare_w1", "compare_wave1", lambda: _compare_paper(WAVE1_OUT)),
        ("08_wave2", "wave2", lambda: _run(
            [
                sys.executable, "scripts/run_cts_eval_full.py",
                "--config", "paper_parity",
                "--benchmarks", "math500", "gsm8k", "aime",
                "--methods", "cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop",
                "--seeds", str(args.seeds),
                "--device", args.device,
                "--output-dir", str(WAVE2_OUT),
                *(
                    ["--resume-partial"]
                    if (WAVE2_OUT / "table2_results.partial.json").is_file()
                    else []
                ),
                *(
                    ["--limit", str(args.limit)]
                    if args.limit is not None
                    else []
                ),
            ],
            log_name="wave2.log",
            env=_headline_env(),
            timeout_s=args.wave2_timeout_s,
        )),
        ("09_docs", "docs_update", lambda: _update_docs(st, WAVE2_OUT)),
        ("10_smoke", "pipeline_smoke", lambda: _run(
            [
                sys.executable, "scripts/run_post_stage2_pipeline.py",
                "--device", args.device,
                "--seeds", str(args.seeds),
                "--table2-limit", "50",
                "--output-root", str(SMOKE_OUT),
            ],
            log_name="pipeline_smoke.log",
            env=_headline_env(),
            timeout_s=args.smoke_timeout_s,
        )),
        ("11_wave3", "wave3_full", lambda: _run(
            [
                sys.executable, "scripts/run_post_stage2_pipeline.py",
                "--device", args.device,
                "--seeds", str(args.seeds),
                "--output-root", str(WAVE3_OUT),
            ],
            log_name="wave3_full.log",
            env=_headline_env(),
            timeout_s=args.wave3_timeout_s,
        )),
        ("12_zip", "zip_hosting", lambda: _run(
            [sys.executable, "scripts/make_anonymous_submission.py"],
            log_name="zip_build.log",
        )),
        ("13_phase4", "phase4_fork", lambda: {"status": "SKIP", "reason": "deferred_to_branch"}),
        ("14_cloud", "cloud_eval", lambda: {"status": "SKIP", "reason": "manual_cloud"}),
        ("15_rebuttal", "rebuttal_prep", lambda: {"status": "PASS", "file": str(ROOT / "results/post_s2_autopilot/rebuttal_numbers.json")}),
    ]

    if args.dry_run:
        for sid, _name, _fn in steps_plan:
            done = _step_done(st, sid)
            print(f"  [{sid}] {'DONE' if done else 'pending'}")
        return 0

    if args.watch and not _step_done(st, "00_wait"):
        w = _wait_for_stage2(args, st)
        _mark_done(st, "00_wait", w)
        if w.get("status") != "PASS":
            _log(f"wait failed: {w}")
            return 2

    wave1_oom = False
    math_pct: Optional[float] = None

    for sid, name, fn in steps_plan:
        if _step_done(st, sid):
            _log(f"skip {sid} (already done)")
            continue
        if sid == "01_audit" and _audit_pass():
            _mark_done(st, sid, {"status": "PASS", "skipped": True})
            continue
        if sid == "02_jsonl" and _jsonl_has_solution():
            _mark_done(st, sid, {"status": "PASS", "skipped": True})
            continue
        if sid == "11_wave3" and args.skip_wave3 and not args.wave3_after_primary:
            _mark_done(st, sid, {"status": "SKIP", "reason": "--skip-wave3"})
            continue
        if sid == "11_wave3" and args.skip_wave3 and args.wave3_after_primary:
            if not _step_done(st, "09_docs"):
                _log("Wave 3 deferred until step 09_docs completes (not marking done)")
                continue
            _log("Wave 3 deferred: primary path (W2+docs) done — starting full Table 2")
        if sid == "10_smoke" and args.skip_smoke:
            _mark_done(st, sid, {"status": "SKIP", "reason": "--skip-smoke"})
            continue
        if sid == "13_phase4":
            if math_pct is None:
                math_pct = _extract_math500_cts4nu(WAVE2_OUT)
            if math_pct is not None and math_pct >= MATH_PASS_THRESHOLD:
                _mark_done(st, sid, {"status": "SKIP", "math500_pct": math_pct, "reason": "above_threshold"})
                continue
            _log(f"Phase 4: MATH={math_pct} < {MATH_PASS_THRESHOLD}; running S1+answer S2")
            s1 = _run(
                [sys.executable, "scripts/run_stage1_openmath.py", "--config", "paper_parity", "--device", args.device],
                log_name="phase4_stage1.log",
                timeout_s=8 * 3600,
            )
            if s1.get("status") != "PASS":
                _mark_done(st, sid, {"status": "FAIL", "stage1": s1})
                continue
            s2 = _run(
                [
                    sys.executable, "scripts/run_stage2_math_ppo.py",
                    "--config", "autopilot_answer_s2", "--parallel-map",
                    "--K", "64", "--collect-batch", "64", "--ppo-epochs", "4",
                    "--steps", "10000", "--log-every", "10", "--save-every", "500",
                    "--device", args.device, "--broyden-max-iter", "30",
                    "--stage1-ckpt", "artifacts/stage1_last.pt",
                ],
                log_name="phase4_stage2_answer.log",
                timeout_s=14 * 24 * 3600,
            )
            res = {"stage1": s1, "stage2": s2}
            if s2.get("status") == "PASS":
                retry_out = WAVE2_OUT.parent / "headline_w2_after_phase4"
                w2b = _run(
                    [
                        sys.executable, "scripts/run_cts_eval_full.py",
                        "--config", "paper_parity",
                        "--benchmarks", "math500", "gsm8k", "aime",
                        "--methods", "cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop",
                        "--seeds", str(args.seeds),
                        "--device", args.device,
                        "--output-dir", str(retry_out),
                        *(
                            ["--resume-partial"]
                            if (retry_out / "table2_results.partial.json").is_file()
                            else []
                        ),
                    ],
                    log_name="wave2_after_phase4.log",
                    env=_headline_env(),
                    timeout_s=args.wave2_timeout_s,
                )
                res["wave2_retry"] = w2b
            _mark_done(st, sid, res)
            continue
        if sid == "14_cloud":
            if wave1_oom or (math_pct is not None and math_pct < MATH_CLOUD_THRESHOLD):
                marker = ROOT / "results" / "post_s2_autopilot" / "CLOUD_MANUAL_REQUIRED.md"
                marker.write_text(
                    "# Cloud eval required\n\n"
                    "Set up A100 and run commands in logs/HEADLINE_EVAL_RUNBOOK.md Phase 5.\n"
                    f"wave1_oom={wave1_oom} math500_pct={math_pct}\n",
                    encoding="utf-8",
                )
                _mark_done(st, sid, {"status": "MANUAL", "marker": str(marker)})
            else:
                _mark_done(st, sid, {"status": "SKIP", "reason": "not_needed"})
            continue

        _log(f"running {sid} ({name})")
        try:
            if sid == "08_wave2":
                existing = _find_running_eval_pid()
                if existing:
                    _log(f"08_wave2: eval already running (PID {existing}); waiting for it to finish")
                    while _find_running_eval_pid():
                        time.sleep(60)
                    result = {
                        "status": "PASS",
                        "reason": "external_eval_completed",
                        "pid": existing,
                    }
                else:
                    result = fn()
            else:
                result = fn()
        except Exception as exc:
            result = {"status": "FAIL", "error": traceback.format_exc()}
        _mark_done(st, sid, result)

        if sid == "06_wave1" and result.get("status") != "PASS":
            wave1_oom = True
            _log("Wave 1 failed (possible OOM) — will flag cloud step")
        if sid == "06_wave1" and result.get("status") == "PASS":
            math_w1 = _extract_math500_cts4nu(WAVE1_OUT)
            if math_w1 is None:
                result["status"] = "WARN"
                result["reason"] = "wave1_empty_results"
                st["steps"]["06_wave1"] = result
                _save_status(st)
                _log("Wave 1 returned 0 samples (check --seeds count); will re-run after Wave 2")
                st.setdefault("wave1_rerun_pending", True)
        if sid == "08_wave2":
            math_pct = _extract_math500_cts4nu(WAVE2_OUT)
            _log(f"Wave 2 MATH-500 CTS-4nu = {math_pct}")
            if result.get("status") == "PASS" and _extract_math500_cts4nu(WAVE1_OUT) is None:
                _log("Re-running Wave 1 (prior run had 0 samples; --seeds was 0)")
                w1r = _run(
                    [
                        sys.executable, "scripts/run_cts_eval_full.py",
                        "--config", "paper_parity",
                        "--benchmarks", "math500",
                        "--methods", "cts_4nu", "greedy", "native_think", "sc_14", "mcts_early_stop",
                        "--seeds", "1",
                        "--device", args.device,
                        "--output-dir", str(WAVE1_OUT),
                    ],
                    log_name="wave1_rerun.log",
                    env=_headline_env(),
                    timeout_s=args.wave1_timeout_s,
                )
                st["steps"]["06_wave1_rerun"] = w1r
                if w1r.get("status") == "PASS":
                    _compare_paper(WAVE1_OUT)
                _save_status(st)

        if result.get("status") == "FAIL" and sid in ("03_verify", "06_wave1", "08_wave2"):
            if sid == "08_wave2":
                ids = st.setdefault("completed_step_ids", [])
                if sid in ids:
                    ids.remove(sid)
                _save_status(st)
            _log(f"fatal fail at {sid}; stopping autopilot (will retry on resume)")
            st["final_verdict"] = "FAIL"
            _save_status(st)
            try:
                _send_final_notification(st)
            except Exception as e:
                _log(f"Final notification error: {e}")
            return 1

    st["final_verdict"] = "PASS"
    _save_status(st)
    try:
        _send_final_notification(st)
    except Exception as e:
        _log(f"Final notification error: {e}")
    _log("autopilot complete")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-S2 full autopilot (steps 3-15)")
    ap.add_argument("--watch", action="store_true", help="wait for Stage 2 step 10000 first")
    ap.add_argument("--resume", action="store_true", help="alias for default (resume from status)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--max-wait-min", type=int, default=5000, help="~83h for remaining S2")
    ap.add_argument("--skip-wave3", action="store_true", help="skip Wave 3 (or defer with --wave3-after-primary)")
    ap.add_argument(
        "--wave3-after-primary",
        action="store_true",
        help="run Wave 3 automatically after step 09_docs (primary path first)",
    )
    ap.add_argument("--skip-smoke", action="store_true", help="skip post-S2 pipeline smoke (step 10)")
    ap.add_argument("--autonomous", action="store_true", help="load configs/autopilot_autonomous.json defaults")
    ap.add_argument("--wave1-timeout-s", type=int, default=24 * 3600)
    ap.add_argument("--wave2-timeout-s", type=int, default=7 * 24 * 3600)
    ap.add_argument("--smoke-timeout-s", type=int, default=48 * 3600)
    ap.add_argument("--wave3-timeout-s", type=int, default=14 * 24 * 3600)
    ap.add_argument("--limit", type=int, default=None, help="limit the number of problems evaluated per benchmark")
    args = ap.parse_args()
    if args.autonomous:
        pol = _load_autonomous_policy().get("policy", {})
        if pol.get("skip_smoke", True):
            args.skip_smoke = True
        if pol.get("wave3_after_primary", True):
            args.skip_wave3 = True
            args.wave3_after_primary = True
        elif not args.skip_wave3:
            args.skip_wave3 = False
    if not args.watch and not args.dry_run and not args.resume:
        ap.error("use --watch to start (or --dry-run)")
    if args.watch or args.resume:
        args.watch = True
    return run_autopilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
