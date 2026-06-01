"""Cognitive Tree Search (CTS) — paper-aligned public API.

This file is the single source of truth for the symbols a paper reviewer can
expect to ``from cts import ...``. Every name in ``__all__`` corresponds to
a paper section listed in ``README.md`` "Paper ↔ Code Mapping".

Heavy backbones (Gemma, Triton kernels, FAISS index training) are NOT
imported here. They live behind PEP-562 lazy attribute access in
:mod:`cts.backbone` so a plain ``import cts`` is a few milliseconds and
runs on a CPU-only reviewer machine.
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- core dataclasses (paper §4.1, §4.3) ---
from cts.types import (
    NuVector,
    NuConfigMode,
    RuntimeBudgetState,
    TransitionResult,
    TreeNode,
)

# --- Algorithm 1 + DEQ transition (paper §4.1-§4.2) ---
from cts.mcts.cts_episode import cts_full_episode
from cts.mcts.puct import puct_score, select_action
from cts.deq.transition import transition, transition_batch

# --- meta-policy / neuro-critic (paper §4.1) ---
from cts.policy.meta_policy import MetaPolicy
from cts.critic.neuro_critic import NeuroCritic

# --- latent context (paper §4.4) ---
from cts.latent.faiss_context import LatentContextWindow

# --- hybrid KV-assist (paper §7.7) ---
from cts.mcts.hybrid_kv import HybridKVManager, hybrid_transition_decision

# --- reward (paper Eq. 5) ---
from cts.rewards.shaping import paper_reward

# --- statistical protocol (paper §7.1) ---
from cts.eval.statistics import (
    bootstrap_ci,
    wilcoxon_signed_rank,
    bonferroni_correct,
)

__all__ = [
    "__version__",
    # core dataclasses
    "NuVector",
    "NuConfigMode",
    "RuntimeBudgetState",
    "TransitionResult",
    "TreeNode",
    # Algorithm 1 + DEQ
    "cts_full_episode",
    "puct_score",
    "select_action",
    "transition",
    "transition_batch",
    # policy / critic
    "MetaPolicy",
    "NeuroCritic",
    # latent context
    "LatentContextWindow",
    # hybrid kv
    "HybridKVManager",
    "hybrid_transition_decision",
    # reward
    "paper_reward",
    # statistics
    "bootstrap_ci",
    "wilcoxon_signed_rank",
    "bonferroni_correct",
]
