#!/usr/bin/env bash
# scripts/upload_checkpoints.sh
#
# Upload the minimum-required CTS NeurIPS 2026 checkpoints from
# the LOCAL machine to a cloud GPU box.
#
# Required uploads (always):
#   - artifacts/stage2_meta_value.pt   (27 MB; meta-policy + critic)
#
# Optional uploads (recommended for full paper-faithful replication):
#   - artifacts/stage1_last.pt         (15 GB; backbone+LoRA delta)
#
# Without stage1_last.pt: the cloud box uses the HF-default Gemma
# weights (no DEQ warm-up). CTS-4nu numbers will be lower than
# the paper headline but still non-zero.
#
# Usage (from your LOCAL machine):
#   bash scripts/upload_checkpoints.sh <ssh-host> [ssh-port]
#   bash scripts/upload_checkpoints.sh root@1.2.3.4 22
#   bash scripts/upload_checkpoints.sh root@<your-cloud-gpu-host> 12345
#
# Or with --full to also upload stage1_last.pt (15 GB, takes 30-60 min):
#   bash scripts/upload_checkpoints.sh --full root@1.2.3.4 22

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
    shift
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/upload_checkpoints.sh [--full] <ssh-host> [ssh-port]" >&2
    echo "  e.g. bash scripts/upload_checkpoints.sh root@1.2.3.4 22" >&2
    echo "  e.g. bash scripts/upload_checkpoints.sh --full root@<cloud-gpu-host> 22" >&2
    exit 2
fi

SSH_HOST="$1"
SSH_PORT="${2:-22}"
REMOTE_PATH="/workspace/cts/artifacts"

# Sanity check
if [[ ! -f "artifacts/stage2_meta_value.pt" ]]; then
    echo "[FATAL] artifacts/stage2_meta_value.pt missing locally." >&2
    echo "        Run training first OR ensure you are at the repo root." >&2
    exit 2
fi

echo "============================================================"
echo "Uploading checkpoints to ${SSH_HOST}:${SSH_PORT}"
echo "============================================================"
echo "  remote dir : ${REMOTE_PATH}"
echo "  mode       : $([[ ${FULL} -eq 1 ]] && echo 'FULL (15 GB)' || echo 'QUICK (27 MB)')"
echo "============================================================"

# Step 1: ensure remote artifacts/ dir exists
echo
echo "[1/3] Creating remote artifacts directory..."
ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_HOST}" \
    "mkdir -p ${REMOTE_PATH}"

# Step 2: upload stage2_meta_value.pt (always)
echo
echo "[2/3] Uploading stage2_meta_value.pt (27 MB)..."
scp -P "${SSH_PORT}" \
    "artifacts/stage2_meta_value.pt" \
    "${SSH_HOST}:${REMOTE_PATH}/stage2_meta_value.pt"

# Step 3: optionally upload stage1_last.pt (15 GB)
if [[ ${FULL} -eq 1 ]]; then
    echo
    echo "[3/3] Uploading stage1_last.pt (15 GB; 30-60 min on typical home upload)..."
    if [[ ! -f "artifacts/stage1_last.pt" ]]; then
        echo "  [WARN] artifacts/stage1_last.pt missing; skipping."
    else
        scp -P "${SSH_PORT}" \
            "artifacts/stage1_last.pt" \
            "${SSH_HOST}:${REMOTE_PATH}/stage1_last.pt"
    fi
else
    echo
    echo "[3/3] Skipping stage1_last.pt (use --full to include)."
fi

# Verify on remote
echo
echo "Verifying remote artifacts:"
ssh -p "${SSH_PORT}" "${SSH_HOST}" \
    "ls -la ${REMOTE_PATH}/ | grep -E 'stage[12]'"

echo
echo "============================================================"
echo "[OK] Upload complete."
echo "============================================================"
echo
echo "Next: SSH into the cloud box and run:"
echo "   ssh -p ${SSH_PORT} ${SSH_HOST}"
echo "   cd /workspace/cts"
echo "   bash scripts/replicate_neurips_2026.sh"
