# Stage 2 watcher + post-stage2 pipeline auto-launcher (Windows / PowerShell)
#
# Polls until the Stage 2 PPO retrain process is gone *and* the
# checkpoint mtime is recent enough to be the post-retrain artefact,
# then immediately invokes scripts/run_post_stage2_pipeline.py with
# paper-faithful flags so D11 night closes Tables 2 + 17 + 19 + ZIP
# rebuild without human intervention.
#
# Usage (typical):
#   powershell -ExecutionPolicy Bypass -File scripts\wait_and_run_pipeline.ps1
#
# Optional flags (forwarded to the pipeline):
#   -Seeds N          (default 5)
#   -Device cuda:0    (default cuda:0)
#   -OutputRoot path  (default results/post_stage2_D11)
#   -StagePid <int>   (PID of the in-flight Stage 2 process; if omitted
#                      we look up the most recent python.exe whose
#                      command line contains run_stage2_math_ppo.py)
#   -PollSeconds N    (default 60)
#   -CkptPath path    (default artifacts/stage2_meta_value.pt)
#
# Exit code: forwarded from run_post_stage2_pipeline.py
#   0 = PASS / PASS_WITH_WARN
#   1 = PARTIAL_FAIL / FAIL_PHASE_1

param(
    [int]    $Seeds         = 5,
    [string] $Device        = "cuda:0",
    [string] $OutputRoot    = "results/post_stage2_D11",
    [int]    $StagePid      = 0,
    [int]    $PollSeconds   = 60,
    [string] $CkptPath      = "artifacts/stage2_meta_value.pt",
    # Maximum time to wait for Stage 2 to finish. Stage-2 PPO at the
    # paper-faithful config runs ~12 GPU-h on a single 4090, so 18 h
    # gives 50 % slack. Past this point the watcher gives up and
    # writes a STAGE2_TIMEOUT marker so a follow-up run can act on it.
    [int]    $MaxWaitMin    = 1080,
    # Heartbeat file: updated each poll so an external operator can
    # see at a glance whether the watcher is still alive.
    [string] $HeartbeatPath = "results/.watcher_heartbeat.json",
    # D-7 partial-save knobs forwarded to run_post_stage2_pipeline.py.
    # Use 0 to mean "no limit" (full benchmark). Compute-limited
    # default mirrors the reviewer-facing canonical command in
    # results/table2/PAPER_VS_LOCAL.md (10 problems / bench for
    # Table 2, 30 problems for Table 17 / aime_90).
    [int]    $Table2Limit   = 0,
    [int]    $Table17Limit  = 0,
    # Skip the Stage 2 ckpt verification phase. Use this when the
    # ckpt is known-good (already validated by an earlier pipeline
    # run) and you only want to refresh Tables 2 / 17 / 19.
    [switch] $SkipVerify
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -Path "$PSScriptRoot\..").Path
Set-Location $root

function Get-Stage2Pid {
    $candidates = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
    if (-not $candidates) { return 0 }
    $hit = $candidates | Where-Object { $_.CommandLine -and $_.CommandLine -match 'run_stage2_math_ppo\.py' } | Sort-Object CreationDate -Descending | Select-Object -First 1
    if ($hit) { return [int]$hit.ProcessId }
    return 0
}

if ($StagePid -le 0) {
    $StagePid = Get-Stage2Pid
    if ($StagePid -le 0) {
        Write-Warning "Stage 2 process not found via CIM; we will only watch the checkpoint mtime."
    } else {
        Write-Host ("[watcher] auto-detected Stage 2 PID = {0}" -f $StagePid)
    }
}

$ckptInitMtime = $null
if (Test-Path $CkptPath) {
    $ckptInitMtime = (Get-Item $CkptPath).LastWriteTime
    Write-Host ("[watcher] initial ckpt mtime = {0}" -f $ckptInitMtime)
} else {
    Write-Host "[watcher] ckpt does not exist yet; waiting for first write..."
}

function Write-Heartbeat {
    param(
        [Parameter(Mandatory=$true)] [hashtable] $State,
        [Parameter(Mandatory=$true)] [string]    $Path
    )
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $payload = $State + @{ timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK") }
    ($payload | ConvertTo-Json -Compress) | Set-Content -Path $Path -Encoding utf8
}

$started = Get-Date
$exitReason = $null
while ($true) {
    $now = Get-Date
    $elapsedMin = [math]::Round(($now - $started).TotalMinutes, 1)

    $alive = $false
    if ($StagePid -gt 0) {
        $proc = Get-Process -Id $StagePid -ErrorAction SilentlyContinue
        $alive = [bool]$proc
    }

    $ckptReady = $false
    $ckptStaleSec = -1
    if (Test-Path $CkptPath) {
        $mtime = (Get-Item $CkptPath).LastWriteTime
        $ckptStaleSec = [int]((Get-Date) - $mtime).TotalSeconds
        if ($null -eq $ckptInitMtime) {
            if (((Get-Date) - $mtime).TotalSeconds -lt 600) {
                $ckptReady = $true
            }
        } elseif ($mtime -gt $ckptInitMtime) {
            $ckptReady = $true
        }
    }

    Write-Heartbeat -Path $HeartbeatPath -State @{
        watcher_pid    = $PID
        stage2_pid     = $StagePid
        stage2_alive   = $alive
        ckpt_path      = $CkptPath
        ckpt_exists    = (Test-Path $CkptPath)
        ckpt_ready     = $ckptReady
        ckpt_stale_sec = $ckptStaleSec
        elapsed_min    = $elapsedMin
        max_wait_min   = $MaxWaitMin
    }

    if ((-not $alive) -and $ckptReady) {
        Write-Host "[watcher] Stage 2 process gone AND ckpt fresh -> launching post-Stage-2 pipeline."
        $exitReason = "OK_LAUNCH_PIPELINE"
        break
    }

    # Crash detection: process gone but ckpt still stale (older than
    # our start time). This means Stage 2 died before saving its final
    # checkpoint -- looping forever would silently miss the deadline.
    if ((-not $alive) -and (-not $ckptReady) -and ($StagePid -gt 0)) {
        Write-Warning "[watcher] Stage 2 process gone but ckpt is stale -- treating as CRASH."
        $marker = Join-Path (Split-Path -Parent $HeartbeatPath) "STAGE2_CRASHED.txt"
        @(
            "Stage 2 PPO retrain process (pid $StagePid) exited without",
            "writing a fresh artifacts/stage2_meta_value.pt.",
            "Last seen alive: ${elapsedMin} min after watcher start.",
            "ckpt mtime is ${ckptStaleSec}s old at detection time.",
            "Recommended action: inspect logs/stage2_full_retrain_*.{log,err}",
            "and restart Stage 2 from artifacts/stage1_last.pt."
        ) | Set-Content -Path $marker -Encoding utf8
        $exitReason = "STAGE2_CRASHED"
        break
    }

    if ($elapsedMin -ge $MaxWaitMin) {
        Write-Warning "[watcher] timeout after $elapsedMin min (max $MaxWaitMin)."
        $marker = Join-Path (Split-Path -Parent $HeartbeatPath) "STAGE2_TIMEOUT.txt"
        @(
            "Stage 2 watcher hit the $MaxWaitMin min ceiling without seeing",
            "the Stage 2 process exit AND a fresh checkpoint.",
            "Either the retrain hung, OOM'd silently, or the ckpt path",
            "drifted ($CkptPath)."
        ) | Set-Content -Path $marker -Encoding utf8
        $exitReason = "STAGE2_TIMEOUT"
        break
    }

    $aliveStr = if ($alive) { "alive" } else { "gone" }
    $ckptStr  = if ($ckptReady) { "fresh" } else { "stale" }
    Write-Host ("[watcher] elapsed={0,6}min  pid={1} {2}  ckpt={3}" -f $elapsedMin, $StagePid, $aliveStr, $ckptStr)
    Start-Sleep -Seconds $PollSeconds
}

if ($exitReason -ne "OK_LAUNCH_PIPELINE") {
    Write-Error "[watcher] not launching pipeline; exit reason = $exitReason"
    exit 2
}

# Final cooldown — give the OS a moment to finish flushing the ckpt file.
Start-Sleep -Seconds 5

Write-Host "[watcher] kicking off scripts/run_post_stage2_pipeline.py ..."
$argList = @(
    "scripts/run_post_stage2_pipeline.py",
    "--seeds", $Seeds,
    "--device", $Device,
    "--output-root", $OutputRoot
)
if ($Table2Limit -gt 0) {
    $argList += @("--table2-limit", $Table2Limit)
    Write-Host ("[watcher] forwarding --table2-limit {0}" -f $Table2Limit)
}
if ($Table17Limit -gt 0) {
    $argList += @("--table17-limit", $Table17Limit)
    Write-Host ("[watcher] forwarding --table17-limit {0}" -f $Table17Limit)
}
if ($SkipVerify) {
    $argList += "--skip-verify"
    Write-Host "[watcher] forwarding --skip-verify (Stage 2 ckpt assumed pre-validated)"
}
$env:CTS_DISABLE_TRITON = "1"
& python @argList
exit $LASTEXITCODE
