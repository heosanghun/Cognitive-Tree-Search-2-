"""
N-ply anchor chain beyond 2-ply: repeated root rollouts → transition → new anchor.

Generalizes `two_ply_mcts_rollouts` to `n_plies` transition edges.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.episode import RootRolloutsResult, mcts_root_rollouts
from cts.mcts.puct import PUCTVariant
from cts.policy.meta_policy import MetaPolicy
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


@dataclass
class MultiPlyRolloutResult:
    """Chain of anchors and transitions; includes rollouts after the last anchor."""

    anchors: List[str]
    transitions: List[TransitionResult]
    rollouts_per_ply: List[RootRolloutsResult] = field(default_factory=list)
    leaf_mean_q: float = 0.0


def multi_ply_mcts_rollouts(
    parent_text: str,
    *,
    n_plies: int = 2,
    sims_per_ply: int = 4,
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
) -> MultiPlyRolloutResult:
    if n_plies < 1:
        raise ValueError("n_plies must be >= 1")
    bb = backbone or MockTinyBackbone(hidden=d, num_layers=42)
    nu_eff = nu or NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    anchors: List[str] = [parent_text]
    transitions: List[TransitionResult] = []
    rollouts: List[RootRolloutsResult] = []
    cur = parent_text
    priors: Optional[List[float]] = None

    for ply in range(n_plies):
        r = mcts_root_rollouts(
            cur,
            num_simulations=sims_per_ply,
            W=W,
            K=K,
            d=d,
            nu=nu_eff,
            priors=priors,
            meta_policy=meta_policy if ply == 0 else None,
            backbone=bb,
            broyden_max_iter=broyden_max_iter,
            tau_flops_budget=tau_flops_budget,
            puct_variant=puct_variant,
            reward_fn=reward_fn,
            faiss_context=faiss_context,
        )
        rollouts.append(r)
        priors = r.priors
        visited = [a for a in range(W) if r.ns[a] > 0]
        best_a = max(visited, key=lambda i: r.qs[i]) if visited else 0
        budget = RuntimeBudgetState()
        tr = transition(
            cur,
            best_a,
            r.nu,
            budget,
            bb,
            K=K,
            d=d,
            broyden_max_iter=broyden_max_iter,
            tau_flops_budget=tau_flops_budget,
            faiss_context=faiss_context,
        )
        transitions.append(tr)
        child_snippet = (tr.child_text or "").strip() or f"<branch {best_a}>"
        cur = f"{cur}{separator}{child_snippet}"
        anchors.append(cur)
        nu_eff = r.nu

    leaf = mcts_root_rollouts(
        cur,
        num_simulations=sims_per_ply,
        W=W,
        K=K,
        d=d,
        nu=nu_eff,
        priors=priors,
        meta_policy=None,
        backbone=bb,
        broyden_max_iter=broyden_max_iter,
        tau_flops_budget=tau_flops_budget,
        puct_variant=puct_variant,
        reward_fn=reward_fn,
        faiss_context=faiss_context,
    )
    rollouts.append(leaf)
    qs_nonzero = [leaf.qs[i] for i in range(W) if leaf.ns[i] > 0]
    leaf_mean_q = sum(qs_nonzero) / max(len(qs_nonzero), 1)

    return MultiPlyRolloutResult(
        anchors=anchors,
        transitions=transitions,
        rollouts_per_ply=rollouts,
        leaf_mean_q=leaf_mean_q,
    )
