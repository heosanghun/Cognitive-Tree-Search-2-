"""Sparse module routing (paper Eq. 6) — reference PyTorch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def routing_weights(
    z: torch.Tensor,
    w_g: torch.Tensor,
    nu_temp: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """alpha = softmax( (W_g @ pool(z)) / νtemp ), z: [K, d], w_g: [n_mod, d]

    Paper §5.2 Eq.(6): routing temperature νtemp applied token-wise.
    """
    pooled = z.mean(dim=0)
    logits = (w_g @ pooled) / max(float(nu_temp), eps)
    return F.softmax(logits, dim=-1)


def top_k_mask(alpha: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, alpha.numel())
    _, idx = torch.topk(alpha, k)
    m = torch.zeros_like(alpha)
    m[idx] = 1.0
    return m


def sparse_module_weights(alpha: torch.Tensor, k: int) -> torch.Tensor:
    """Renormalize over top-k for weighted sum (reference path)."""
    m = top_k_mask(alpha, k)
    w = alpha * m
    return w / (w.sum().clamp_min(1e-8))
