# Live Wave 2 progress dashboard (refreshes every 3s).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\watch_wave2_progress.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\watch_wave2_progress.ps1 -Interval 5

param(
    [double]$Interval = 2
)

$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
# Seed PID cache for watch script (optional; watch is log-based by default).
$pidFile = Join-Path $root "results\post_s2_autopilot\watch_pids.json"
$eval = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'run_cts_eval_full\.py' } | Select-Object -First 1
$ap = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'post_s2_autopilot\.py' } | Select-Object -First 1
if ($eval -or $ap) {
    @{ eval_pid = $(if ($eval) { $eval.ProcessId } else { $null })
       autopilot_pid = $(if ($ap) { $ap.ProcessId } else { $null })
       updated_at = (Get-Date -Format o) } | ConvertTo-Json | Set-Content $pidFile -Encoding utf8
}

python -u scripts/watch_wave2_progress.py --watch --interval $Interval --scan-pids
