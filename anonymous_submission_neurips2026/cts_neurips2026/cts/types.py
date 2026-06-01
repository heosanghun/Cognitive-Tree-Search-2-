"""Core CTS datatypes aligned with paper §4.1.

Paper §4.1: nu = [nu_expl, nu_tol, nu_temp, nu_act] in R^4.
V(z*) is separate Neuro-Critic output, NOT part of nu.

Paper Table 5: nu-component Pareto configurations:
  CTS-4nu: all active {expl, tol, temp, act}
  CTS-2nu: {expl, temp} active; tol, act fixed at Stage 1 converged means
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set

import torch


# Paper Table 5: nu-configuration Pareto modes
NuConfigMode = Literal["4nu", "3nu_no_act", "2nu_expl_tol", "2nu_fast", "1nu"]

NU_CONFIG_ACTIVE: Dict[NuConfigMode, FrozenSet[str]] = {
    "4nu": frozenset({"expl", "tol", "temp", "act"}),
    "3nu_no_act": frozenset({"expl", "tol", "temp"}),
    "2nu_expl_tol": frozenset({"expl", "tol"}),
    "2nu_fast": frozenset({"expl", "temp"}),
    "1nu": frozenset({"expl"}),
}

# Stage 1 converged means (paper Table 5 footnote)
NU_STAGE1_DEFAULTS: Dict[str, float] = {
    "tol": 5e-3,
    "act": 0.78,
    "temp": 1.0,
    "expl": 1.0,
}


@dataclass
class NuVector:
    """Meta-policy output per step (paper §4.1).

    nu = [nu_expl, nu_tol, nu_temp, nu_act] in R^4.
    nu_val is kept for backward compatibility but is NOT part of the
    paper's nu vector — V(z*) comes from separate Neuro-Critic.
    """

    nu_val: float = 1.0
    nu_expl: float = 1.0
    nu_tol: float = 0.5
    nu_temp: float = 1.0
    nu_act: float = 1.0

    def apply_config(self, mode: NuConfigMode) -> "NuVector":
        """Apply nu-config mode: fix inactive operators to Stage 1 means.

        Paper Table 5: "Fixed nu: set to Stage 1 converged mean; no inference
        overhead. All configurations exceed Native Think (42.5%). No retraining
        required for mode switching."
        """
        active = NU_CONFIG_ACTIVE[mode]
        return NuVector(
            nu_val=self.nu_val,
            nu_expl=self.nu_expl if "expl" in active else NU_STAGE1_DEFAULTS["expl"],
            nu_tol=self.nu_tol if "tol" in active else NU_STAGE1_DEFAULTS["tol"],
            nu_temp=self.nu_temp if "temp" in active else NU_STAGE1_DEFAULTS["temp"],
            nu_act=self.nu_act if "act" in active else NU_STAGE1_DEFAULTS["act"],
        )

    @property
    def nu_da(self) -> float:
        return self.nu_val

    @property
    def nu_5ht(self) -> float:
        return self.nu_expl

    @property
    def nu_ne(self) -> float:
        return self.nu_tol

    @property
    def nu_ach(self) -> float:
        return self.nu_temp

    @property
    def nu_ado_scale(self) -> float:
        return self.nu_act


@dataclass
class RuntimeBudgetState:
    """Environment-held compute accumulation (paper §4.3 ACT)."""

    mac_accumulated: float = 0.0
    terminal_depth: int = 0
    flops_spent_step: float = 0.0
    wall_clock_ms_step: float = 0.0

    @property
    def ado_accumulated(self) -> float:
        return self.mac_accumulated

    def clone(self) -> "RuntimeBudgetState":
        return RuntimeBudgetState(
            mac_accumulated=self.mac_accumulated,
            terminal_depth=self.terminal_depth,
            flops_spent_step=self.flops_spent_step,
            wall_clock_ms_step=self.wall_clock_ms_step,
        )


@dataclass
class TransitionResult:
    child_text: Optional[str]
    z_star_child: torch.Tensor
    solver_stats: Dict[str, Any]
    prune: bool
    budget: RuntimeBudgetState
    faiss_retrieved: Optional[torch.Tensor] = None


@dataclass
class MCTSStats:
    visit_count: int = 0
    q_value: float = 0.0
    prior: float = 0.0


@dataclass
class TreeNode:
    """Discrete search node with hard-anchored text state s_t (paper)."""

    text_state: str
    z_star: Optional[torch.Tensor]
    depth: int
    parent_id: Optional[int]
    node_id: int
    children_ids: List[int] = field(default_factory=list)
    mcts_N: int = 0
    mcts_W: int = 3
    mcts_Q: List[float] = field(default_factory=list)
    mcts_prior: List[float] = field(default_factory=list)
    # Paper Remark 2 (Jacobian Inheritance): converged inverse Jacobian from
    # solving the DEQ at this node, threaded into child solves as their warm
    # start. Populated only on the dense-Broyden path (n <= MAX_DENSE_N);
    # remains None on the Anderson path used by full Gemma-scale tensors.
    inv_jacobian_state: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not self.mcts_Q:
            self.mcts_Q = [0.0] * self.mcts_W
        if not self.mcts_prior:
            self.mcts_prior = [1.0 / self.mcts_W] * self.mcts_W
