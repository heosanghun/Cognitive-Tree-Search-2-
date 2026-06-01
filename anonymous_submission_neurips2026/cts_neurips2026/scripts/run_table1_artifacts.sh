#!/usr/bin/env bash
# Table 1 style artifacts — run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts

echo "== CTS mock + analytic KV + latency =="
python -m cts.eval.profile_vram_latency --depths 1 5 10 15 20 --out artifacts/table1_profile.csv "$@"

echo "== KV measured (CUDA) =="
python scripts/profile_kv_measured.py --depths 1 5 10 15 --out artifacts/table1_kv_measured.csv

echo "Done. Outputs written to artifacts/."
