# Push a single-line training status file to Git (run hourly via Task Scheduler).
# GitHub cloud runners cannot read your local log — this must run on the training PC.
#
# Usage (once):
#   Set environment variables or edit defaults below, then test:
#   powershell -ExecutionPolicy Bypass -File scripts\push_training_status_one_line.ps1
#
# Env (optional):
#   CTS_STATUS_LOG      — full path to stage2 log (default: latest stage2_paper_full*.log under repo/logs)
#   CTS_STATUS_REPO     — git repo root (default: parent of scripts/)
#   CTS_STATUS_REL_PATH — path inside repo for the one-line file (default: training_status_one_line.txt)

param(
    [string]$LogPath = $env:CTS_STATUS_LOG,
    [string]$RepoRoot = $env:CTS_STATUS_REPO,
    [string]$RelativeStatusFile = $(if ($env:CTS_STATUS_REL_PATH) { $env:CTS_STATUS_REL_PATH } else { "training_status_one_line.txt" })
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if (-not $LogPath) {
    $logsDir = Join-Path $RepoRoot "logs"
    if (-not (Test-Path $logsDir)) {
        Write-Error "No CTS_STATUS_LOG and no logs directory: $logsDir"
    }
    $latest = Get-ChildItem -Path $logsDir -Filter "stage2_paper_full*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        Write-Error "No stage2_paper_full*.log under $logsDir — set CTS_STATUS_LOG explicitly."
    }
    $LogPath = $latest.FullName
}

if (-not (Test-Path $LogPath)) {
    Write-Error "Log not found: $LogPath"
}

# Last stage2 progress line (log-every 10 may repeat; take last match).
$tail = Get-Content -Path $LogPath -Tail 800 -ErrorAction Stop
$matchLine = ($tail | Select-String -Pattern "stage2 step=\d+/\d+" | Select-Object -Last 1).Line
if (-not $matchLine) {
    $matchLine = ($tail | Select-String -Pattern "stage2 step=" | Select-Object -Last 1).Line
}
if (-not $matchLine) {
    $matchLine = "(no stage2 step line in last 800 lines)"
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
$oneLine = "${ts} | ${matchLine}"

$statusPath = Join-Path $RepoRoot $RelativeStatusFile
# UTF-8 without BOM (PS 5.1 safe)
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($statusPath, $oneLine + "`r`n", $utf8)

Push-Location $RepoRoot
try {
    git add -- $RelativeStatusFile
    $diff = git diff --cached --quiet 2>$null; $hasChange = $LASTEXITCODE -ne 0
    if (-not $hasChange) {
        Write-Host "No change to $RelativeStatusFile — skip commit/push."
        exit 0
    }
    git commit -m "chore: training status (hourly) [skip ci]"
    git push
    Write-Host "Pushed: $oneLine"
}
finally {
    Pop-Location
}
