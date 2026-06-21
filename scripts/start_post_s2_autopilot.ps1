# Detached launcher for post_s2_autopilot.py (steps 3-15 after Stage 2).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_post_s2_autopilot.ps1
#
# Logs:
#   logs/post_s2_autopilot_launcher.log
#   results/post_s2_autopilot/logs/autopilot.log
#   results/post_s2_autopilot/autopilot_status.json

# Avoid duplicate autopilot processes (use -ForceRestart to kill stale stack)
param(
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$launcherLog = Join-Path $logDir "post_s2_autopilot_launcher.log"
$statusPath  = Join-Path $root "results\post_s2_autopilot\autopilot_status.json"

$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'post_s2_autopilot\.py' }
if ($existing -and -not $ForceRestart) {
    Write-Host "[launcher] autopilot already running (PID $($existing[0].ProcessId))"
    exit 0
}
if ($existing -and $ForceRestart) {
    foreach ($p in $existing) {
        Write-Host "[launcher] ForceRestart: stopping autopilot PID $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    $evalProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'run_cts_eval_full\.py' }
    foreach ($p in $evalProcs) {
        Write-Host "[launcher] ForceRestart: stopping eval PID $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

$argList = @(
    "-u", "scripts/post_s2_autopilot.py",
    "--resume",
    "--autonomous",
    "--skip-smoke",
    "--device", "cuda:0",
    "--seeds", "5",
    "--limit", "50"
)

Write-Host "[launcher] starting detached post_s2_autopilot.py ..."
Write-Host "[launcher] status -> $statusPath"

$outLog = Join-Path $logDir "post_s2_autopilot_stdout.log"
$errLog = Join-Path $logDir "post_s2_autopilot_stderr.log"

$proc = Start-Process -FilePath "python" `
    -ArgumentList $argList `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru

Write-Host "[launcher] autopilot PID = $($proc.Id)"
Write-Host "[launcher] tail: Get-Content results\post_s2_autopilot\logs\autopilot.log -Wait -Tail 5"
