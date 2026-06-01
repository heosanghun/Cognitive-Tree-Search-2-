"""Value head on z* — Neuro-Critic νval (paper §5.3).

Implemented as a separate linear head atop the shared Meta-Policy backbone.
V(z*) is computed directly from the latent space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class NeuroCritic(nn.Module):
    """Paper §5.3: outputs νval = V(z*) directly from the universal latent space."""

    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.Tanh(), nn.Linear(z_dim, 1)
        )

    def forward(self, z_star_flat: torch.Tensor) -> torch.Tensor:
        """z_star_flat: [batch, z_dim] or [z_dim] → scalar value(s)."""
        if z_star_flat.dim() == 1:
            z_star_flat = z_star_flat.unsqueeze(0)
        return self.net(z_star_flat)

    def batch_evaluate(self, z_star_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate W branches simultaneously (paper §4.1).

        z_star_batch: [W, K, d] → mean-pool → [W, d] → V(z*): [W, 1]
        """
        W = z_star_batch.shape[0]
        pooled = z_star_batch.mean(dim=1)  # [W, d]
        d = pooled.shape[-1]
        z_dim = self.net[0].in_features
        if d != z_dim:
            if d > z_dim:
                pooled = pooled[:, :z_dim]
            else:
                padded = torch.zeros(W, z_dim, device=pooled.device, dtype=pooled.dtype)
                padded[:, :d] = pooled
                pooled = padded
        return self.net(pooled)
