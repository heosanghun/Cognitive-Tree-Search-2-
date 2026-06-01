"""Global RNG seed from CTS_GLOBAL_SEED for reproducible runs."""

from __future__ import annotations

import os
import random
from typing import Optional


def apply_global_seed() -> Optional[int]:
    """
    If env CTS_GLOBAL_SEED is set to an integer, seed Python, NumPy (if installed), and torch.
    Call once at process start of training/eval entry points.
    """
    raw = os.environ.get("CTS_GLOBAL_SEED", "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except ValueError:
        return None
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
