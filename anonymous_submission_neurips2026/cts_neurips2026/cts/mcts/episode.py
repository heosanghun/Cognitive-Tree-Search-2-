"""
Shallow MCTS-style expansion: root node + W `transition()` calls (mock or real backbone).

Also: **PUCT select one child → single `transition`** (short loop) with optional ν / priors
from `MetaPolicy` (paper §4.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.deq.transition import transition
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.puct import PUCTVariant, select_action
from cts.mcts.tree import SearchTree
from cts.policy.meta_policy import MetaPolicy
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


def default_transition_reward(r: TransitionResult) -> float:
    """1.0 if DEQ converged and not pruned; else 0.0."""
    if r.prune:
        return 0.0
    return 1.0 if r.solver_stats.get("converged") else 0.0


def parent_text_features(parent_text: str, dim: int = 64) -> torch.Tensor:
    """Deterministic bag-of-chars embedding for `MetaPolicy` (no Gemma encode)."""
    v = torch.zeros(dim, dtype=torch.float32)
    for i, c in enumerate(parent_text[:512]):
        v[i % dim] += float(ord(c)) / 1024.0
    return v


def expand_root_parallel_branches(
    parent_text: str,
    *,
    W: int = 3,
    K: int = 64,
    d: int = 32,
    nu: NuVector | None = None,
    backbone: object | None = None,
    broyden_max_iter: int = 30,
    tau_flops_budget: float = 1e14,
    faiss_context: Optional[LatentContextWindow] = None,
) -> Tuple[SearchTree, List[TransitionResult]]:
    """
    Create root, run `transition` for branch_index in [0, W) — paper branching factor W.
    """
    bb = backbone or MockTinyBackbone(hidden=d, num_layers=42)
    nu = nu or NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    tree = SearchTree()
    root_id = tree.new_node(parent_text, None, 0, None, W=W)
    results: List[TransitionResult] = []
    for b in range(W):
        budget = RuntimeBudgetState()
        r = transition(
            parent_text,
            b,
            nu,
            budget,
            bb,
            K=K,
            d=d,
            broyden_max_iter=broyden_max_iter,
            tau_flops_budget=tau_flops_budget,
            faiss_context=faiss_context,
        )
        results.append(r)
        child_text = r.child_text or ""
        tree.new_node(child_text, r.z_star_child, 1, root_id, W=W)
    return tree, results


@dataclass
class PUCTExpandOnceResult:
    tree: SearchTree
    transition: TransitionResult
    selected_action: int
    nu: NuVector
    priors: List[float]


def puct_select_and_expand_once(
    parent_text: str,
    *,
    W: int = 3,
    K: int = 64,
    d: int = 32,
    nu: NuVector | None = None,
    priors: List[float] | None = None,
    meta_policy: Optional[MetaPolicy] = None,
    text_features: torch.Tensor | None = None,
    backbone: object | None = None,
    broyden_max_iter: int = 30,
    tau_flops_budget: float = 1e14,
    puct_variant: PUCTVariant = "paper",
    n_root_visits: int = 0,
    reward_fn: Optional[Callable[[TransitionResult], float]] = None,
    faiss_context: Optional[LatentContextWindow] = None,
) -> PUCTExpandOnceResult:
    """
    At root: PUCT picks one action `a`, then a single `transition(..., branch_index=a)`.
    """
    bb = backbone or MockTinyBackbone(hidden=d, num_layers=42)
    if meta_policy is not None:
        dim = int(meta_policy.enc.in_features)
        feats = (
            text_features
            if text_features is not None
            else parent_text_features(parent_text, dim=dim)
        )
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        nu, priors = meta_policy(feats)
    else:
        if nu is None:
            nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
        if priors is None:
            priors = [1.0 / W] * W

    ns = [0] * W
    qs = [0.0] * W
    n_parent = max(1, n_root_visits)
    a = select_action(puct_variant, nu.nu_expl, priors, ns, qs, n_parent)

    budget = RuntimeBudgetState()
    r = transition(
        parent_text,
        a,
        nu,
        budget,
        bb,
        K=K,
        d=d,
        broyden_max_iter=broyden_max_iter,
        tau_flops_budget=tau_flops_budget,
        faiss_context=faiss_context,
    )
    reward = reward_fn(r) if reward_fn is not None else default_transition_reward(r)
    qs[a] = reward
    ns[a] = 1

    tree = SearchTree()
    root_id = tree.new_node(parent_text, None, 0, None, W=W)
    tree.nodes[root_id].mcts_N = 1
    tree.nodes[root_id].mcts_prior = list(priors)
    tree.nodes[root_id].mcts_Q = list(qs)
    child_text = r.child_text or ""
    tree.new_node(child_text, r.z_star_child, 1, root_id, W=W)

    return PUCTExpandOnceResult(
        tree=tree,
        transition=r,
        selected_action=a,
        nu=nu,
        priors=list(priors),
    )


@dataclass
class RootRolloutsResult:
    """After `num_simulations` root-level PUCT selections + transitions + mean-Q backup."""

    ns: List[int]
    qs: List[float]
    nu: NuVector
    priors: List[float]
    tree: SearchTree
    transitions: List[TransitionResult] = field(default_factory=list)


def mcts_root_rollouts(
    parent_text: str,
    *,
    num_simulations: int = 4,
    W: int = 3,
    K: int = 64,
    d: int = 32,
    nu: NuVector | None = None,
    priors: List[float] | None = None,
    meta_policy: Optional[MetaPolicy] = None,
    text_features: torch.Tensor | None = None,
    backbone: object | None = None,
    broyden_max_iter: int = 30,
    tau_flops_budget: float = 1e14,
    puct_variant: PUCTVariant = "paper",
    reward_fn: Optional[Callable[[TransitionResult], float]] = None,
    faiss_context: Optional[LatentContextWindow] = None,
) -> RootRolloutsResult:
    """
    Repeat `num_simulations` times: PUCT at root → one `transition` → **backup** mean Q(a).
    """
    bb = backbone or MockTinyBackbone(hidden=d, num_layers=42)
    if meta_policy is not None:
        dim = int(meta_policy.enc.in_features)
        feats = (
            text_features
            if text_features is not None
            else parent_text_features(parent_text, dim=dim)
        )
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        nu, priors = meta_policy(feats)
    else:
        if nu is None:
            nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
        if priors is None:
            priors = [1.0 / W] * W

    ns = [0] * W
    qs = [0.0] * W
    transitions: List[TransitionResult] = []

    for _ in range(num_simulations):
        n_parent = max(1, sum(ns))
        a = select_action(puct_variant, nu.nu_expl, priors, ns, qs, n_parent)
        budget = RuntimeBudgetState()
        r = transition(
            parent_text,
            a,
            nu,
            budget,
            bb,
            K=K,
            d=d,
            broyden_max_iter=broyden_max_iter,
            tau_flops_budget=tau_flops_budget,
            faiss_context=faiss_context,
        )
        transitions.append(r)
        rew = reward_fn(r) if reward_fn is not None else default_transition_reward(r)
        old_n = ns[a]
        new_n = old_n + 1
        qs[a] = (qs[a] * old_n + rew) / float(new_n)
        ns[a] = new_n

    tree = SearchTree()
    root_id = tree.new_node(parent_text, None, 0, None, W=W)
    tree.nodes[root_id].mcts_N = sum(ns)
    tree.nodes[root_id].mcts_Q = list(qs)
    tree.nodes[root_id].mcts_prior = list(priors)

    return RootRolloutsResult(
        ns=list(ns),
        qs=list(qs),
        nu=nu,
        priors=list(priors),
        tree=tree,
        transitions=transitions,
    )
