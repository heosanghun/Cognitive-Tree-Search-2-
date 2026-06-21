# Kill stale Wave 2 stack and restart autonomous pipeline (autopilot + watchdog).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\restart_wave2_pipeline.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

Write-Host "[restart] healing status + killing stale processes ..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.post_s2_autopilot import _load_status, _heal_incomplete_wave2
st = _load_status()
_heal_incomplete_wave2(st)
print('completed:', st.get('completed_step_ids'))
"

& "$PSScriptRoot\start_post_s2_autopilot.ps1" -ForceRestart

$existingWd = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'autopilot_watchdog\.py' }
foreach ($p in $existingWd) {
    Write-Host "[restart] stopping old watchdog PID $($p.ProcessId)"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

$wdLog = Join-Path $root "logs\autopilot_watchdog_stdout.log"
$wdErr = Join-Path $root "logs\autopilot_watchdog_stderr.log"
$wd = Start-Process -FilePath "python" `
    -ArgumentList @("-u", "scripts/autopilot_watchdog.py", "--watch") `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $wdLog `
    -RedirectStandardError $wdErr `
    -PassThru
Write-Host "[restart] watchdog PID = $($wd.Id)"
Write-Host "[restart] monitor: python scripts/watch_wave2_progress.py --watch"
