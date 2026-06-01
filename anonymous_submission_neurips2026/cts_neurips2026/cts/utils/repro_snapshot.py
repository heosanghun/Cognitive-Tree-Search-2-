"""Write environment snapshot JSON for experiment provenance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _git_head(root: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def collect_repro_dict(*, root: Path | None = None) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    out: Dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "repo_root": str(root),
        "git_commit": _git_head(root),
        "cts_global_seed": os.environ.get("CTS_GLOBAL_SEED"),
        "cts_gemma_model_dir": os.environ.get("CTS_GEMMA_MODEL_DIR"),
        "cts_deq_map_mode": os.environ.get("CTS_DEQ_MAP_MODE"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["cuda_version"] = torch.version.cuda
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        out["torch"] = None
    return out


def write_repro_snapshot(path: Path, *, root: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = collect_repro_dict(root=root)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
