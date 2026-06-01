"""Load YAML configs; optional shallow merge with default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(
    name: str = "default",
    configs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    root = configs_dir or Path(__file__).resolve().parents[2] / "configs"
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if name != "default":
        default_path = root / "default.yaml"
        if default_path.is_file():
            with default_path.open("r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
            data = _deep_merge(base, data)
    return data
