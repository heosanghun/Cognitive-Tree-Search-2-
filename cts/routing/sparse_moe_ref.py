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


def sparse_module_weights(
    alpha: torch.Tensor, k: int, renormalize: bool = False
) -> torch.Tensor:
    """Sparse top-k module weights (paper Eq. 3).

    Paper Eq. 3 sums the *raw* softmax weights over the Top-k modules:
    ``z* = sum_{i in Top-k} Softmax(W_g z*/nu_temp)_i · m_i(...)`` — no
    renormalization. The resulting weights sum to < 1, which also serves
    Proposition 1's contraction condition ``sum_i g_i L_i < 1``.

    ``renormalize=True`` restores the pre-alignment behaviour (weights
    rescaled to sum to 1) for backward comparison.
    """
    m = top_k_mask(alpha, k)
    w = alpha * m
    if renormalize:
        return w / (w.sum().clamp_min(1e-8))
    return w
