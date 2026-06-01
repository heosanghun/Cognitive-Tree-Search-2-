"""UCB1 bandit over discretised nu_expl (paper Table 2, Bandit row).

Paper: 20 arms, exploration coefficient c = sqrt(2), gradient-free
adaptive control of nu_expl at inference time (no PPO retraining).

CPU-friendly: pure Python + math; no torch required for the bandit itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class UCB1NuExplBandit:
    """Discretise ``[nu_min, nu_max]`` into ``n_arms`` bins; UCB1 selection."""

    n_arms: int = 20
    c: float = math.sqrt(2.0)
    nu_min: float = 0.1
    nu_max: float = 2.0
    _counts: List[int] = field(default_factory=list)
    _values: List[float] = field(default_factory=list)
    _total: int = 0

    def __post_init__(self) -> None:
        if self.n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if self.nu_max <= self.nu_min:
            raise ValueError("nu_max must exceed nu_min")
        self._counts = [0] * self.n_arms
        self._values = [0.0] * self.n_arms

    def arm_values(self) -> List[float]:
        """Centre of each discretised nu_expl bin."""
        if self.n_arms == 1:
            return [(self.nu_min + self.nu_max) / 2.0]
        step = (self.nu_max - self.nu_min) / float(self.n_arms - 1)
        return [self.nu_min + i * step for i in range(self.n_arms)]

    def select(self) -> tuple[int, float]:
        """Return ``(arm_index, nu_expl_value)`` for the next episode."""
        for arm in range(self.n_arms):
            if self._counts[arm] == 0:
                return arm, self.arm_values()[arm]
        log_t = math.log(max(self._total, 1))
        best_arm = 0
        best_ucb = -float("inf")
        for arm in range(self.n_arms):
            mean = self._values[arm] / self._counts[arm]
            bonus = self.c * math.sqrt(log_t / self._counts[arm])
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        return best_arm, self.arm_values()[best_arm]

    def update(self, arm: int, reward: float) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm out of range: {arm}")
        self._counts[arm] += 1
        self._values[arm] += float(reward)
        self._total += 1

    def snapshot(self) -> dict:
        return {
            "n_arms": self.n_arms,
            "c": self.c,
            "counts": list(self._counts),
            "values": list(self._values),
            "total": self._total,
        }
