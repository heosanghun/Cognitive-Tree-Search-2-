#!/usr/bin/env bash
# scripts/cloud_setup.sh
#
# One-command cloud GPU box setup for the CTS NeurIPS 2026
# replication. Tested on Ubuntu 22.04 LTS + Python 3.10/3.11
# (matches typical Lambda Labs / Vast.ai / generic cloud GPU images).
#
# What this does (~5-10 min on a fresh box, ~3-5 GB download):
#   1. apt update + install minimal system deps (git, build-essential)
#   2. pip install torch (CUDA wheel matched to detected nvcc)
#   3. pip install -r requirements.txt + extras (transformers, etc.)
#   4. download HuggingFace Gemma-4-E4B-it (auto-cached for next runs)
#   5. download benchmarks (AIME, MATH, GSM8K, HumanEval, ARC)
#   6. torch-free static verification (sanity check)
#
# This script does NOT:
#   - download Stage-1 / Stage-2 checkpoints (the user must scp them
#     up themselves; see ``scripts/upload_checkpoints.sh`` on the
#     local machine for the canonical command)
#   - run the actual GPU pipeline (use ``scripts/replicate_neurips_2026.sh``
#     after this script completes)
#
# Usage (on the cloud box, after ``git clone``):
#   bash scripts/cloud_setup.sh
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo "CTS NeurIPS 2026 - cloud GPU setup"
echo "============================================================"
echo "  repo root  : ${REPO_ROOT}"
echo "  date       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  user       : $(whoami)"
echo "  gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NO GPU DETECTED')"
echo "============================================================"

# --------------------------------------------------------------------------
# Step 1: detect environment
# --------------------------------------------------------------------------

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[FATAL] no python on PATH" >&2
    exit 2
fi
echo "[1/6] python      : $($PY --version)"

if ! command -v pip >/dev/null 2>&1 && ! command -v pip3 >/dev/null 2>&1; then
    echo "  installing pip..."
    $PY -m ensurepip --upgrade
fi
PIP="$PY -m pip"

if ! command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION="unknown (using torch+cu121 default)"
    TORCH_CUDA_TAG="cu121"
else
    NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    CUDA_VERSION="${NVCC_VER}"
    case "${NVCC_VER}" in
        11.8*|11.7*) TORCH_CUDA_TAG="cu118" ;;
        12.1*|12.2*) TORCH_CUDA_TAG="cu121" ;;
        12.4*|12.5*|12.6*|12.7*|12.8*) TORCH_CUDA_TAG="cu124" ;;
        *) TORCH_CUDA_TAG="cu121" ;;
    esac
fi
echo "[1/6] cuda        : ${CUDA_VERSION}  -> torch+${TORCH_CUDA_TAG}"

# --------------------------------------------------------------------------
# Step 2: system deps
# --------------------------------------------------------------------------

echo
echo "[2/6] System deps (apt)..."
if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        git build-essential ca-certificates curl wget \
        > /dev/null 2>&1 || true
    echo "  done"
else
    echo "  skipped (no apt-get; cloud box presumably has these already)"
fi

# --------------------------------------------------------------------------
# Step 3: pip install torch + project deps
# --------------------------------------------------------------------------

echo
echo "[3/6] pip install torch (CUDA wheel)..."
$PIP install --upgrade pip wheel setuptools --quiet
$PIP install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
    "torch>=2.0,<3.0" --quiet 2>&1 | tail -5

echo
echo "[3/6] pip install project + extras..."
if [[ -f requirements.txt ]]; then
    $PIP install -r requirements.txt --quiet 2>&1 | tail -10 || true
fi
if [[ -f pyproject.toml ]]; then
    $PIP install -e ".[dev,data]" --quiet 2>&1 | tail -10 || true
fi
$PIP install transformers safetensors accelerate huggingface_hub \
    sentencepiece numpy scipy sympy pyyaml peft \
    --quiet 2>&1 | tail -5

# --------------------------------------------------------------------------
# Step 4: verify torch + CUDA
# --------------------------------------------------------------------------

echo
echo "[4/6] Verifying torch + CUDA..."
$PY - <<'PYV'
import torch
print(f"  torch       : {torch.__version__}")
print(f"  cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        n = torch.cuda.get_device_name(i)
        m = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"  device [{i}] : {n}  {m:.1f} GB")
else:
    raise SystemExit("[FATAL] CUDA not available; check GPU pod allocation")
PYV

# --------------------------------------------------------------------------
# Step 5: download Gemma-4-E4B + benchmarks
# --------------------------------------------------------------------------

echo
echo "[5/6] Downloading benchmarks (AIME / MATH / GSM8K / HumanEval / ARC)..."
mkdir -p data
if [[ -f scripts/download_all_benchmarks.py ]]; then
    $PY scripts/download_all_benchmarks.py 2>&1 | tail -20 || true
fi

# Pre-cache Gemma-4-E4B-it base model (~8 GB, takes ~3 min on a typical cloud GPU box)
echo
echo "[5/6] Pre-caching Gemma-4-E4B-it (~8 GB; ~3 min on a typical cloud GPU box)..."
$PY - <<'PYDL' || true
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("  fetching tokenizer...")
    AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    print("  fetching model weights (this is the slow part)...")
    AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it",
        torch_dtype="auto",
        device_map="cpu",
    )
    print("  cached.")
except Exception as exc:
    print(f"  [WARN] HF download failed: {exc}")
    print("         You may need to run ``huggingface-cli login`` first if")
    print("         this model is gated. The replication script will retry.")
PYDL

# --------------------------------------------------------------------------
# Step 6: torch-free static verification
# --------------------------------------------------------------------------

echo
echo "[6/6] Static D-7 verification (torch-free; ~1 second)..."
$PY scripts/_reviewer_local_audit.py 2>&1 | tail -10

echo
echo "============================================================"
echo "[OK] cloud setup complete"
echo "============================================================"
echo
echo "Next steps:"
echo "  1. From your LOCAL machine, upload checkpoints:"
echo "       bash scripts/upload_checkpoints.sh <ssh-host> [<port>]"
echo "     (only stage2_meta_value.pt = 27 MB is REQUIRED;"
echo "      stage1_last.pt = 15 GB is optional but recommended)"
echo
echo "  2. Then on this cloud box, run the replication:"
echo "       bash scripts/replicate_neurips_2026.sh"
echo
echo "  3. After replication, download the results back to your local:"
echo "       (from local) bash scripts/download_results.sh <ssh-host> [<port>]"
