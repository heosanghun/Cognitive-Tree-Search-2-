"""Latency / VRAM sweep utilities for Table-1 style reports."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from cts.perf.memory_accounting import cuda_peak_marker, peak_allocated_bytes


def run_timed(fn: Callable[[], Any], cuda_sync: bool = True) -> tuple[Any, float]:
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def write_sweep_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
