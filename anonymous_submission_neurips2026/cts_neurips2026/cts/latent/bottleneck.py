"""Latent bottleneck: z0 init, exploration noise, and Wproj decoding (paper §4.1, §4.5)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def init_z0(
    K: int, d: int, device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    g = generator or torch.Generator(device=device)
    z = torch.randn(K, d, generator=g, dtype=torch.float32, device=device) * 0.02
    return z


def add_exploration_noise(
    z0: torch.Tensor, nu_expl: float, generator: torch.Generator
) -> torch.Tensor:
    """Paper §4.1: νexpl controls noise variance injected into z0 before DEQ solve."""
    sigma = 0.05 * float(nu_expl)
    noise = torch.randn(
        z0.shape, dtype=z0.dtype, device=z0.device, generator=generator
    )
    return z0 + sigma * noise


# Legacy alias
def add_serotonin_noise(
    z0: torch.Tensor, nu_5ht: float, generator: torch.Generator
) -> torch.Tensor:
    return add_exploration_noise(z0, nu_5ht, generator)


class LatentProjection(nn.Module):
    """Wproj: Latent-to-Text Decoding Projection (paper §4.5).

    Projects K latent tokens into continuous soft prompt that bypasses
    '<|think|>' format and maps directly to [Final Answer] semantic space.

    Wproj ∈ R^{d × dmodel}
    """

    def __init__(self, latent_dim: int, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, model_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, z_star: torch.Tensor) -> torch.Tensor:
        """Project z* [K, d] → soft prompt [K, dmodel]."""
        return self.proj(z_star)


class LatentDecoder(nn.Module):
    """Complete latent-to-text decoder combining Wproj with frozen model decode.

    Paper §4.5: terminal text generation via one-time standard AR overhead.
    """

    def __init__(
        self,
        latent_dim: int,
        model_dim: int,
        vocab_size: int,
        *,
        max_decode_tokens: int = 500,
    ) -> None:
        super().__init__()
        self.wproj = LatentProjection(latent_dim, model_dim)
        self.output_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.max_decode_tokens = max_decode_tokens

    def project_to_soft_prompt(self, z_star: torch.Tensor) -> torch.Tensor:
        return self.wproj(z_star)

    def greedy_logits(self, z_star: torch.Tensor) -> torch.Tensor:
        """Get logits from projected z* (for testing without full AR decode)."""
        soft_prompt = self.wproj(z_star)
        return self.output_head(soft_prompt)


def validate_information_retention(
    z_star: torch.Tensor,
    decoder: LatentDecoder,
    reference_tokens: torch.Tensor,
    *,
    threshold: float = 0.914,
) -> dict:
    """Paper Appendix H: validate K=64 retains ≥91.4% symbolic info.

    Returns dict with match_rate and pass/fail status.
    """
    with torch.no_grad():
        logits = decoder.greedy_logits(z_star)
        predicted = logits.argmax(dim=-1)
        K = min(predicted.shape[0], reference_tokens.shape[0])
        matches = (predicted[:K] == reference_tokens[:K]).float().mean().item()
    return {
        "match_rate": matches,
        "threshold": threshold,
        "passed": matches >= threshold,
    }
