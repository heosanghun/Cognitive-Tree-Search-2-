"""Bridge `NeuroCritic` to `reward_fn(TransitionResult)` for MCTS rollouts."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from cts.types import TransitionResult


def z_star_to_vector(z: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Mean-pool then pad/trim to `out_dim`."""
    v = z.mean(dim=0).reshape(-1).float()
    if v.numel() >= out_dim:
        return v[:out_dim]
    out = torch.zeros(out_dim, dtype=v.dtype, device=v.device)
    out[: v.numel()] = v
    return out


def make_critic_reward_fn(
    critic: nn.Module,
    *,
    z_dim: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> Callable[[TransitionResult], float]:
    """
    Returns scalar reward in ~[0,1] from sigmoid(V(z*)).
    """
    dev = device or next(critic.parameters()).device
    crit = critic.to(dev)
    crit.eval()

    @torch.inference_mode()
    def _fn(r: TransitionResult) -> float:
        if r.z_star_child is None:
            return 0.0
        x = z_star_to_vector(r.z_star_child, z_dim).to(dev)
        v = crit(x.unsqueeze(0)).squeeze()
        return float(torch.sigmoid(v / max(temperature, 1e-6)).item())

    return _fn
