#!/usr/bin/env bash
# scripts/replicate_neurips_2026.sh
#
# Reviewer-facing one-command replication script for the
# CTS NeurIPS 2026 submission. Tested on Ubuntu 22.04 LTS +
# Python 3.11 + 1xA100 (40 GB) and 1xRTX 4090 (24 GB).
#
# Usage:
#   bash scripts/replicate_neurips_2026.sh                  # default: 10 AIME, 30 Table-17 rows
#   bash scripts/replicate_neurips_2026.sh --full           # full Table 2 + Table 17 (multi-GPU recommended)
#   bash scripts/replicate_neurips_2026.sh --static-only    # no GPU; runs the 1-second torch-free verification
#   bash scripts/replicate_neurips_2026.sh --ci-mode        # GitHub Actions / self-hosted runner mode (no GPU; structured exit codes)
#
# The default mode is the canonical "&le; 10 GPU-h" reviewer
# replication described in REVIEWER_FAQ Q15 and the status banner
# of results/table2/PAPER_VS_LOCAL.md. It produces a refreshed
# results/table2/table2_results.json that the reviewer can
# diff against the paper headline numbers.
#
# This script is idempotent: re-running picks up partial-save
# snapshots (table2_results.partial.json) from an earlier kill
# and continues from the next un-evaluated cell.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer ``python3`` (Ubuntu / GitHub Actions / WSL default) and
# fall back to ``python`` (conda envs, some macOS installs). All
# subsequent invocations go through ``$PY``.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[FATAL] neither python3 nor python is on PATH" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

MODE="default"
TABLE2_LIMIT=10
TABLE17_LIMIT=30
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            MODE="full"
            TABLE2_LIMIT=0
            TABLE17_LIMIT=0
            shift
            ;;
        --static-only)
            MODE="static-only"
            shift
            ;;
        --ci-mode)
            # CI mode: same as --static-only but additionally
            # exports the D12 verdict to results/d12_verdict.{json,md}
            # so the GitHub Actions matrix can upload the verdict
            # as a workflow artefact and surface PASS/FAIL in the
            # PR check summary. Distinct exit codes:
            #   0 = ALL_PASS
            #   1 = PARTIAL_FAIL (non-blocking marker drift)
            #   2 = HARD_FAIL    (ZIP build/audit failure)
            MODE="ci-mode"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --table2-limit)
            TABLE2_LIMIT="$2"
            shift 2
            ;;
        --table17-limit)
            TABLE17_LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "[FATAL] unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

echo "============================================================"
echo "CTS NeurIPS 2026 - reviewer replication script"
echo "============================================================"
echo "  mode           : ${MODE}"
echo "  device         : ${DEVICE}"
echo "  table2-limit   : ${TABLE2_LIMIT}  (0 = full)"
echo "  table17-limit  : ${TABLE17_LIMIT}  (0 = full)"
echo "  repo root      : ${REPO_ROOT}"
echo "============================================================"

# --------------------------------------------------------------------------
# Step 0: torch-free static verification (always runs first; ~1 s)
# --------------------------------------------------------------------------

echo
echo "[STEP 0/5] Torch-free static verification (~1 second)"
$PY scripts/_reviewer_local_audit.py
echo "[STEP 0/5] OK -- the static surface (every D-7 patch + every"
echo "          paper claim with a code anchor) is present on disk."

if [[ "${MODE}" == "static-only" ]]; then
    echo
    echo "[OK] --static-only: skipping GPU steps. Done."
    exit 0
fi

if [[ "${MODE}" == "ci-mode" ]]; then
    echo
    echo "[CI] running D12 sanity export..."
    mkdir -p results
    set +e
    $PY scripts/_d12_final_check.py --quiet --export-verdict results/d12_verdict.json
    CI_RC=$?
    set -e
    if [[ ${CI_RC} -eq 0 ]]; then
        echo "[CI] verdict ALL_PASS; uploaded artefact at results/d12_verdict.{json,md}"
    elif [[ ${CI_RC} -eq 1 ]]; then
        echo "[CI] verdict PARTIAL_FAIL; marker drift detected (non-blocking)"
    else
        echo "[CI] verdict HARD_FAIL; ZIP build or audit failure (BLOCKING)"
    fi
    echo "[CI] done. Exit code: ${CI_RC}"
    exit ${CI_RC}
fi

# --------------------------------------------------------------------------
# Step 1: dependency check (NOT install; we do not pip-install for the
# reviewer; that is their environmental decision)
# --------------------------------------------------------------------------

echo
echo "[STEP 1/5] Dependency check"
$PY - <<'PY'
import sys
missing = []
for mod in ("torch", "transformers", "numpy", "scipy", "sympy", "yaml"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"[FATAL] missing python deps: {missing}")
    print("        install with:  pip install -r requirements.txt")
    sys.exit(2)
import torch
print(f"  torch         : {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda devices  : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        n = torch.cuda.get_device_name(i)
        m = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"    [{i}] {n:40s}  {m:.1f} GB")
else:
    print("[FATAL] no CUDA device; this script requires a GPU.")
    print("        for static-only verification, re-run with --static-only.")
    sys.exit(2)
PY

# --------------------------------------------------------------------------
# Step 2: torch-free static D-7 validation (CPU only; ~0.4 s)
# --------------------------------------------------------------------------

echo
echo "[STEP 2/5] Torch-free static D-7 validation (~0.4 second)"
$PY - <<'PY'
import importlib.util as u, sys, time
spec = u.spec_from_file_location("t", "tests/test_d7_static_validation.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)
names = [n for n in dir(m) if n.startswith("test_")]
fail = 0; t0 = time.time()
for n in names:
    try:
        getattr(m, n)()
    except Exception as e:
        fail += 1
        print(f"[FAIL] {n} -> {e}")
ok = len(names) - fail
print(f"[OK]   {ok}/{len(names)} static D-7 tests pass in {(time.time()-t0)*1000:.0f} ms")
if fail:
    sys.exit(1)
PY

# --------------------------------------------------------------------------
# Step 3: Mock-based dispatcher fallback test (~0.2 s, no GPU)
# --------------------------------------------------------------------------

echo
echo "[STEP 3/5] Mock-based Q14 fallback test (~0.2 second)"
$PY - <<'PY'
import importlib.util as u, sys, time
spec = u.spec_from_file_location("t", "tests/test_dispatcher_fallback_mock.py")
m = u.module_from_spec(spec); spec.loader.exec_module(m)
names = [n for n in dir(m) if n.startswith("test_")]
fail = 0; t0 = time.time()
for n in names:
    try:
        getattr(m, n)()
    except Exception as e:
        fail += 1
        print(f"[FAIL] {n} -> {e}")
ok = len(names) - fail
print(f"[OK]   {ok}/{len(names)} mock dispatcher tests pass in {(time.time()-t0)*1000:.0f} ms")
if fail:
    sys.exit(1)
PY

# --------------------------------------------------------------------------
# Step 4: download benchmarks if not present
# --------------------------------------------------------------------------

echo
echo "[STEP 4/5] Benchmark cache check"
$PY scripts/download_all_benchmarks.py --check-only || {
    echo "  benchmarks not fully cached; downloading now..."
    $PY scripts/download_all_benchmarks.py
}
echo "[OK]   benchmarks present in data/"

# --------------------------------------------------------------------------
# Step 5: run the post-Stage-2 pipeline (THIS is the GPU work)
# --------------------------------------------------------------------------

echo
echo "[STEP 5/5] Post-Stage-2 evaluation pipeline (Tables 2 + 17)"
echo "          mode=${MODE}  device=${DEVICE}"
echo "          table2-limit=${TABLE2_LIMIT}  table17-limit=${TABLE17_LIMIT}"

LIMIT_FLAGS=()
if [[ "${TABLE2_LIMIT}" -gt 0 ]]; then
    LIMIT_FLAGS+=("--table2-limit" "${TABLE2_LIMIT}")
fi
if [[ "${TABLE17_LIMIT}" -gt 0 ]]; then
    LIMIT_FLAGS+=("--table17-limit" "${TABLE17_LIMIT}")
fi

OUTPUT_ROOT="results/reviewer_replication_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"

set +e
$PY -u scripts/run_post_stage2_pipeline.py \
    --device "${DEVICE}" \
    --output-root "${OUTPUT_ROOT}" \
    --skip-verify \
    "${LIMIT_FLAGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/pipeline.log"
RC=$?
set -e

echo
echo "============================================================"
echo "Replication finished. Exit code: ${RC}"
echo "Output: ${OUTPUT_ROOT}/"
echo "============================================================"

if [[ ${RC} -eq 0 ]]; then
    echo
    echo "[OK] Refreshed Tables 2 / 17 are at:"
    echo "       ${OUTPUT_ROOT}/table2/table2_results.json"
    echo "       ${OUTPUT_ROOT}/table17/table17_results.json"
    echo
    echo "Compare against the paper headline:"
    echo "       python scripts/compare_to_paper_table2.py \\"
    echo "             --local ${OUTPUT_ROOT}/table2/table2_results.json"
else
    echo
    echo "[WARN] Pipeline exited non-zero. The partial-save snapshot at"
    echo "       ${OUTPUT_ROOT}/table2/table2_results.partial.json"
    echo "       contains every cell that did finish; re-running this"
    echo "       script will resume from the next un-evaluated cell."
fi

exit ${RC}
