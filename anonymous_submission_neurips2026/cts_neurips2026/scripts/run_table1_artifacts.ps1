# Table 1 style artifacts on Windows — run from repo root with GPU optional.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

Write-Host "== CTS mock + analytic KV + latency (cuda if available) =="
python -m cts.eval.profile_vram_latency --depths 1 5 10 15 20 --out artifacts/table1_profile.csv @args

Write-Host "== KV measured (requires CUDA; else null peaks) =="
python scripts/profile_kv_measured.py --depths 1 5 10 15 --out artifacts/table1_kv_measured.csv

Write-Host "Done. Outputs written to artifacts/."
