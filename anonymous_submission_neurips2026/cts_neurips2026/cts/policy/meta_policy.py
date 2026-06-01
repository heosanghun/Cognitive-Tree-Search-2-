"""Meta-policy: nu vector + branch priors (paper §4.1).

Paper §4.1: "Meta-policy pi_phi: a separate 2-layer MLP outputting
nu = [nu_expl, nu_tol, nu_temp, nu_act] in R^4"

V(z*) is output by Neuro-Critic V_psi, NOT part of nu.
Input: mean-pooled z*_s from the selected MCTS node.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from cts.types import NuVector


class MetaPolicy(nn.Module):
    """Paper §4.1: 2-layer MLP outputting nu in R^4 + branch priors.

    nu = [nu_expl, nu_tol, nu_temp, nu_act]  (4 outputs, NOT 5)
    V(z*) is handled by separate NeuroCritic, not part of nu.
    """

    def __init__(self, text_dim: int = 64, hidden: int = 256, W: int = 3) -> None:
        super().__init__()
        self.W = W
        self.enc = nn.Linear(text_dim, hidden)
        self.act = nn.ReLU()
        self.head_nu = nn.Linear(hidden, 4)
        self.head_prior = nn.Linear(hidden, W)

    def logits_and_nu(
        self, z_star_pooled: torch.Tensor
    ) -> Tuple[NuVector, torch.Tensor]:
        """Returns (nu vector, branch logits tensor) for PPO log-prob.

        Args:
            z_star_pooled: mean-pooled z* from selected node [batch, d] or [d].
        """
        if z_star_pooled.dim() == 1:
            z_star_pooled = z_star_pooled.unsqueeze(0)
        h = self.act(self.enc(z_star_pooled))
        raw = self.head_nu(h).squeeze(0)
        nu = NuVector(
            nu_expl=float(torch.nn.functional.softplus(raw[0]).item()) + 0.5,
            nu_tol=float(torch.sigmoid(raw[1]).item()),
            nu_temp=float(torch.nn.functional.softplus(raw[2]).item()) + 0.5,
            nu_act=float(torch.nn.functional.softplus(raw[3]).item()) + 0.5,
        )
        logits = self.head_prior(h).squeeze(0)
        return nu, logits

    def forward(
        self, z_star_pooled: torch.Tensor
    ) -> Tuple[NuVector, List[float]]:
        nu, logits = self.logits_and_nu(z_star_pooled)
        p = torch.softmax(logits, dim=-1).tolist()
        return nu, p
