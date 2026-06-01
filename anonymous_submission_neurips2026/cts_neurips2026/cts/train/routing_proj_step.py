"""
Stage 2 stub: single optimizer step on `routing_proj` (paper W_g).

- **Mock** (`MockRoutingOnly`): no Gemma — shape [19, d] for pipeline tests.
- **Real**: `GemmaCTSBackbone` loaded with full weights.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MockRoutingOnly(nn.Module):
    """Minimal stand-in with `routing_proj` + `routing_matrix()` for unit tests."""

    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.routing_proj = nn.Parameter(torch.randn(19, d) * 0.02)

    def routing_matrix(self) -> torch.Tensor:
        return self.routing_proj


def routing_target_alignment_loss(
    z: torch.Tensor,
    backbone: nn.Module,
    *,
    nu_temp: float = 1.0,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match softmax(W_g @ pool(z) / νtemp) to `target` (default: uniform 1/19)."""
    w_g = backbone.routing_matrix()
    if target is None:
        target = torch.full(
            (w_g.shape[0],), 1.0 / w_g.shape[0], device=z.device, dtype=torch.float32
        )
    pooled = z.mean(dim=0).float()
    logits = (w_g.float() @ pooled) / max(float(nu_temp), 1e-6)
    p = F.softmax(logits, dim=-1)
    return F.mse_loss(p, target.float())


def routing_entropy(
    z: torch.Tensor, backbone: nn.Module, *, nu_temp: float = 1.0
) -> torch.Tensor:
    """Shannon entropy H(alpha) for alpha = softmax(W_g @ pool(z) / νtemp)."""
    w_g = backbone.routing_matrix()
    pooled = z.mean(dim=0).float()
    logits = (w_g.float() @ pooled) / max(float(nu_temp), 1e-6)
    p = F.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-9).log())).sum()


def routing_loss_paper_style(
    z: torch.Tensor,
    backbone: nn.Module,
    *,
    nu_temp: float = 1.0,
    target: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """MSE to target + optional entropy_coef * H(alpha)."""
    base = routing_target_alignment_loss(z, backbone, nu_temp=nu_temp, target=target)
    if entropy_coef == 0.0:
        return base
    h = routing_entropy(z, backbone, nu_temp=nu_temp)
    return base + entropy_coef * h


def train_routing_proj_one_step(
    backbone: nn.Module,
    *,
    z: torch.Tensor | None = None,
    nu_temp: float = 1.0,
    lr: float = 1e-2,
    device: torch.device | None = None,
    entropy_coef: float = 0.0,
) -> Tuple[float, nn.Module]:
    if not hasattr(backbone, "routing_matrix") or not hasattr(
        backbone, "routing_proj"
    ):
        raise TypeError(
            "backbone must expose routing_matrix() and routing_proj Parameter"
        )
    dev = device or next(backbone.parameters()).device
    if z is None:
        h = int(backbone.routing_proj.shape[1])
        z = torch.randn(8, h, device=dev)
    opt = torch.optim.Adam([backbone.routing_proj], lr=lr)
    opt.zero_grad()
    loss = routing_loss_paper_style(
        z, backbone, nu_temp=nu_temp, entropy_coef=entropy_coef
    )
    loss.backward()
    opt.step()
    return float(loss.detach().cpu().item()), backbone
