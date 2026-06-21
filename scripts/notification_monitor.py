#!/usr/bin/env python3
import os
import sys
import json
import time
import urllib.request
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "notifications.json"
STATUS_PATH = ROOT / "results" / "post_s2_autopilot" / "autopilot_status.json"
MONITOR_STATE_PATH = ROOT / "results" / "post_s2_autopilot" / "monitor_state.json"

def log_local(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_dir = ROOT / "results" / "post_s2_autopilot" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "monitor.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def load_config():
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log_local(f"Error parsing config file: {e}")
    return {
        "discord": {"enabled": false, "webhook_url": ""},
        "slack": {"enabled": false, "webhook_url": ""},
        "telegram": {"enabled": false, "bot_token": "", "chat_id": ""},
        "local_log": {"enabled": True, "log_path": "results/post_s2_autopilot/logs/notifications.log"}
    }

def load_monitor_state():
    if MONITOR_STATE_PATH.is_file():
        try:
            return json.loads(MONITOR_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"notified_step_ids": [], "final_notified": False}

def save_monitor_state(state):
    MONITOR_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

def send_http_post(url: str, data: dict, headers: dict = None) -> bool:
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status in (200, 201, 204)
    except Exception as e:
        log_local(f"Error sending HTTP request to {url}: {e}")
        return False

def dispatch_notification(config, text: str):
    # Local Log
    local_conf = config.get("local_log", {})
    if local_conf.get("enabled", True):
        log_path = ROOT / local_conf.get("log_path", "results/post_s2_autopilot/logs/notifications.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- NOTIFICATION {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{text}\n\n")
        log_local(f"Logged notification locally to {log_path.name}")
        
    # Discord
    discord_conf = config.get("discord", {})
    if discord_conf.get("enabled", False) and discord_conf.get("webhook_url"):
        log_local("Sending Discord notification...")
        send_http_post(discord_conf["webhook_url"], {"content": text})
        
    # Slack
    slack_conf = config.get("slack", {})
    if slack_conf.get("enabled", False) and slack_conf.get("webhook_url"):
        log_local("Sending Slack notification...")
        send_http_post(slack_conf["webhook_url"], {"text": text})
        
    # Telegram
    tg_conf = config.get("telegram", {})
    if tg_conf.get("enabled", False) and tg_conf.get("bot_token") and tg_conf.get("chat_id"):
        log_local("Sending Telegram notification...")
        url = f"https://api.telegram.org/bot{tg_conf['bot_token']}/sendMessage"
        send_http_post(url, {
            "chat_id": tg_conf["chat_id"],
            "text": text
        })

def format_step_message(step_id: str, info: dict) -> str:
    status = info.get("status", "UNKNOWN")
    duration = info.get("duration_s")
    
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
            
    if "reason" in info:
        msg += f"• Reason: {info['reason']}\n"
    if "error" in info:
        msg += f"• Error: {info['error']}\n"
        
    if step_id == "09_docs" and "numbers" in info:
        nums = info["numbers"]
        msg += "• Results Summary:\n"
        for k, v in nums.items():
            if v is not None:
                msg += f"  - {k}: {v}\n"
                
    return msg

def format_final_message(status_data: dict) -> str:
    verdict = status_data.get("final_verdict", "UNKNOWN")
    emoji = "🎉" if verdict == "PASS" else "⚠️"
    msg = f"[{emoji} Autopilot Final Verdict: {verdict}]\n"
    msg += f"• Updated: {status_data.get('updated_at_utc', 'unknown')} UTC\n"
    completed = status_data.get("completed_step_ids", [])
    msg += f"• Completed Steps: {', '.join(completed)}\n"
    
    # Extract Wave 2 numbers if available
    w2_info = status_data.get("steps", {}).get("09_docs", {}).get("numbers", {})
    if w2_info:
        msg += "• Final Evaluation metrics:\n"
        for k, v in w2_info.items():
            if v is not None:
                msg += f"  - {k}: {v}\n"
    return msg

def check_and_notify(config, state, status_data):
    completed_steps = status_data.get("completed_step_ids", [])
    notified_steps = state.setdefault("notified_step_ids", [])
    
    # 1. Check individual steps
    for step_id in completed_steps:
        if step_id not in notified_steps:
            step_info = status_data.get("steps", {}).get(step_id, {})
            msg = format_step_message(step_id, step_info)
            dispatch_notification(config, msg)
            notified_steps.append(step_id)
            save_monitor_state(state)
            
    # 2. Check final verdict
    if "final_verdict" in status_data and not state.get("final_notified", False):
        msg = format_final_message(status_data)
        dispatch_notification(config, msg)
        state["final_notified"] = True
        save_monitor_state(state)

def run_once(config, state):
    if not STATUS_PATH.is_file():
        log_local("Autopilot status file not found yet.")
        return
    try:
        status_data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log_local(f"Error reading status file: {e}")
        return
    check_and_notify(config, state, status_data)

def main():
    parser = argparse.ArgumentParser(description="Notification monitor for post_s2_autopilot")
    parser.add_argument("--once", action="store_true", help="Run check once and exit")
    parser.add_argument("--watch", action="store_true", help="Run in monitoring loop")
    parser.add_argument("--poll-sec", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--test-trigger", action="store_true", help="Send a test notification to verify settings")
    args = parser.parse_args()

    config = load_config()
    state = load_monitor_state()

    if args.test_trigger:
        test_msg = "[🔔 Test Notification] Cognitive-Tree-Search Notification System is successfully configured!"
        dispatch_notification(config, test_msg)
        print("Test trigger notification sent.")
        return 0

    if args.once:
        run_once(config, state)
        return 0

    if args.watch:
        log_local("Starting notification monitor daemon in watch mode...")
        # Check initially
        run_once(config, state)
        try:
            while True:
                time.sleep(args.poll_sec)
                # reload config in case it changes
                config = load_config()
                run_once(config, state)
        except KeyboardInterrupt:
            log_local("Notification monitor stopped by user.")
        return 0

    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())
