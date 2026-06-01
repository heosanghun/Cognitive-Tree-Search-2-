# Detached launcher for post_s2_autopilot.py (steps 3-15 after Stage 2).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start_post_s2_autopilot.ps1
#
# Logs:
#   logs/post_s2_autopilot_launcher.log
#   results/post_s2_autopilot/logs/autopilot.log
#   results/post_s2_autopilot/autopilot_status.json

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$launcherLog = Join-Path $logDir "post_s2_autopilot_launcher.log"
$statusPath  = Join-Path $root "results\post_s2_autopilot\autopilot_status.json"

# Avoid duplicate autopilot processes
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'post_s2_autopilot\.py' }
if ($existing) {
    Write-Host "[launcher] autopilot already running (PID $($existing[0].ProcessId))"
    exit 0
}

$argList = @(
    "-u", "scripts/post_s2_autopilot.py",
    "--watch",
    "--device", "cuda:0",
    "--seeds", "5"
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
