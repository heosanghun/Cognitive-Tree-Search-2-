# Post-S2 autopilot status (auto-updated)

## Autonomous mode: ON

Policy: `configs/autopilot_autonomous.json`  
Launcher: `scripts/start_autonomous_pipeline.ps1` (autopilot + watchdog)

| Mechanism | Purpose |
|-----------|---------|
| `post_s2_autopilot.py --autonomous` | W2 → W1 rerun → docs → zip → Phase4 (if needed) → Wave3 (after primary) → rebuttal |
| `autopilot_watchdog.py --watch` | Restart if eval log idle >20 min (ignores stale autopilot PID) |
| No user prompts | All steps chained; Cloud (14) = marker file only |

## Monitor

```powershell
python scripts/watch_wave2_progress.py --watch
Get-Content results\post_s2_autopilot\logs\watchdog.log -Wait -Tail 5
```
