"""M1/M2 memory accounting helpers (integrate with torch.cuda.max_memory_allocated)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch


@contextmanager
def cuda_peak_marker(device: Optional[torch.device] = None) -> Iterator[None]:
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    try:
        yield
    finally:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)


def peak_allocated_bytes(device: Optional[torch.device] = None) -> int:
    if not torch.cuda.is_available():
        return 0
    dev = device or torch.device("cuda")
    return int(torch.cuda.max_memory_allocated(dev))
