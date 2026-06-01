"""
Two-ply (depth-2) MCTS-style rollouts: root statistics → best child state → second root rollouts.

Uses string concatenation for `parent_text` at ply 2 (paper: discrete text anchor s_t).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.episode import RootRolloutsResult, mcts_root_rollouts
from cts.mcts.puct import PUCTVariant
from cts.policy.meta_policy import MetaPolicy
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


@dataclass
class TwoPlyResult:
    root: RootRolloutsResult
    child: RootRolloutsResult
    best_action: int
    child_anchor_text: str


def two_ply_mcts_rollouts(
    parent_text: str,
    *,
    sims_root: int = 4,
    sims_child: int = 4,
    W: int = 3,
    K: int = 64,
    d: int = 32,
    nu: NuVector | None = None,
    meta_policy: Optional[MetaPolicy] = None,
    backbone: object | None = None,
    separator: str = "\n---\n",
    broyden_max_iter: int = 30,
    tau_flops_budget: float = 1e14,
    puct_variant: PUCTVariant = "paper",
    reward_fn: Optional[Callable[[TransitionResult], float]] = None,
    faiss_context: Optional[LatentContextWindow] = None,
) -> TwoPlyResult:
    bb = backbone or MockTinyBackbone(hidden=d, num_layers=42)
    r1 = mcts_root_rollouts(
        parent_text,
        num_simulations=sims_root,
        W=W,
        K=K,
        d=d,
        nu=nu,
        meta_policy=meta_policy,
        backbone=bb,
        broyden_max_iter=broyden_max_iter,
        tau_flops_budget=tau_flops_budget,
        puct_variant=puct_variant,
        reward_fn=reward_fn,
        faiss_context=faiss_context,
    )

    visited = [a for a in range(W) if r1.ns[a] > 0]
    best_a = max(visited, key=lambda i: r1.qs[i]) if visited else 0

    nu_eff = r1.nu
    budget = RuntimeBudgetState()
    tr = transition(
        parent_text,
        best_a,
        nu_eff,
        budget,
        bb,
        K=K,
        d=d,
        broyden_max_iter=broyden_max_iter,
        tau_flops_budget=tau_flops_budget,
        faiss_context=faiss_context,
    )
    child_snippet = (tr.child_text or "").strip() or f"<branch {best_a}>"
    anchor = f"{parent_text}{separator}{child_snippet}"

    r2 = mcts_root_rollouts(
        anchor,
        num_simulations=sims_child,
        W=W,
        K=K,
        d=d,
        nu=nu_eff,
        priors=r1.priors,
        meta_policy=None,
        backbone=bb,
        broyden_max_iter=broyden_max_iter,
        tau_flops_budget=tau_flops_budget,
        puct_variant=puct_variant,
        reward_fn=reward_fn,
        faiss_context=faiss_context,
    )

    return TwoPlyResult(root=r1, child=r2, best_action=best_a, child_anchor_text=anchor)
