"""Stage 2: PPO-style update on MetaPolicy after mock rollouts (full Gemma+PPO is separate)."""

from __future__ import annotations

from typing import Tuple

import torch

from cts.policy.meta_policy import MetaPolicy


def run_mini_ppo_step(
    meta: MetaPolicy,
    *,
    obs: torch.Tensor,
    old_action: int,
    advantage: float,
    lr: float = 1e-2,
) -> Tuple[float, MetaPolicy]:
    """Single policy-gradient style step on branch head (smoke test)."""
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    opt = torch.optim.Adam(meta.parameters(), lr=lr)
    opt.zero_grad()
    h = meta.act(meta.enc(obs))
    logits = meta.head_prior(h).squeeze(0)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(torch.tensor(old_action))
    loss = -float(advantage) * logp
    loss.backward()
    opt.step()
    return float(loss.detach().cpu().item()), meta


def run_stage2_stub() -> None:
    raise NotImplementedError(
        "Use run_mini_ppo_step for unit tests; full run: python scripts/run_stage2_math_ppo.py"
    )
