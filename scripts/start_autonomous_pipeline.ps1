# Full autonomous post-S2 pipeline: autopilot + watchdog (no user prompts).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_autonomous_pipeline.ps1
#
# Policy: configs/autopilot_autonomous.json
# Status: results/post_s2_autopilot/autopilot_status.json
# Watchdog log: results/post_s2_autopilot/logs/watchdog.log

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

Write-Host "[autonomous] mode=full_autonomous (no user prompts)"
Write-Host "[autonomous] policy -> configs\autopilot_autonomous.json"

# 1) Autopilot (Wave 2 -> docs -> zip -> phase4 -> wave3 deferred -> rebuttal)
& "$PSScriptRoot\start_post_s2_autopilot.ps1"

# 2) Watchdog (restart on eval/autopilot crash)
$existingWd = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'autopilot_watchdog\.py' }
if ($existingWd) {
    Write-Host "[autonomous] watchdog already running (PID $($existingWd[0].ProcessId))"
} else {
    $wdLog = Join-Path $root "logs\autopilot_watchdog_stdout.log"
    $wdErr = Join-Path $root "logs\autopilot_watchdog_stderr.log"
    $wd = Start-Process -FilePath "python" `
        -ArgumentList @("-u", "scripts/autopilot_watchdog.py", "--watch") `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $wdLog `
        -RedirectStandardError $wdErr `
        -PassThru
    Write-Host "[autonomous] watchdog PID = $($wd.Id)"
}

Write-Host "[autonomous] progress: python scripts/watch_wave2_progress.py --watch"
Write-Host "[autonomous] tail: Get-Content results\post_s2_autopilot\logs\watchdog.log -Wait -Tail 5"
