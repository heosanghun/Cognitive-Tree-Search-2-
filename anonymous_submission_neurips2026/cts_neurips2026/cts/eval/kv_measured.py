"""
Measured GPU allocation for a **simplified KV cache** tensor block (Table 1 style).

Allocates K/V per layer with shape [1, num_kv_heads, seq, head_dim] in bf16 to mirror
GQA storage order of magnitude. Not a full transformer forward — **peak VRAM** from
retained tensors only (MCTS path retention proxy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from cts.baselines.mcts_kv_baseline import KVRetentionConfig


def measure_kv_cache_peak_bytes(
    tree_depth: int,
    cfg: KVRetentionConfig | None = None,
    *,
    device: Optional[torch.device] = None,
) -> Optional[int]:
    """
    Returns peak CUDA bytes after allocating layer KV blocks, or None if no CUDA.
    `seq_len = tree_depth * tokens_per_depth_step` (same convention as analytic baseline).
    """
    if not torch.cuda.is_available():
        return None
    # Prime CUDA context — on some Windows/WDDM builds, reset_peak_memory_stats(device)
    # fails before any allocation (Invalid device argument).
    _ = torch.empty(1, device="cuda")
    dev = device or torch.device("cuda")
    cfg = cfg or KVRetentionConfig()
    seq = max(1, int(tree_depth)) * cfg.tokens_per_depth_step

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    except RuntimeError:
        return None
    blocks: List[torch.Tensor] = []
    dtype = torch.bfloat16
    try:
        for _ in range(cfg.num_hidden_layers):
            k = torch.empty(
                1,
                cfg.num_key_value_heads,
                seq,
                cfg.head_dim,
                device=dev,
                dtype=dtype,
            )
            v = torch.empty_like(k)
            blocks.append(k)
            blocks.append(v)
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    except RuntimeError:
        return None


def measure_kv_peak_gb(
    tree_depth: int,
    cfg: KVRetentionConfig | None = None,
    *,
    device: Optional[torch.device] = None,
) -> Optional[float]:
    b = measure_kv_cache_peak_bytes(tree_depth, cfg, device=device)
    if b is None:
        return None
    return b / 1e9


def sweep_kv_measured_rows(
    depths: List[int],
    cfg: KVRetentionConfig | None = None,
    *,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for d in depths:
        gb = measure_kv_peak_gb(d, cfg, device=device)
        rows.append(
            {
                "tree_depth_proxy": d,
                "approach": "kv_tensor_measured",
                "peak_vram_gb": round(gb, 4) if gb is not None else None,
                "notes": "bf16 K/V tensors only; seq = depth * tokens_per_depth_step",
            }
        )
    return rows
