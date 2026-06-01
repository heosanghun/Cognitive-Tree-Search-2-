"""PUCT selection (paper Eq. 4): νexpl replaces static c=1.0."""

from __future__ import annotations

import math
from typing import List, Literal

PUCTVariant = Literal["paper", "alphazero"]


def puct_score(
    variant: PUCTVariant,
    nu_expl: float,
    prior: float,
    n_parent: int,
    n_sa: int,
    q_sa: float,
    c_puct: float = 1.0,
) -> float:
    """
    Paper Eq.(4): U(s,a) = νexpl * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    where P(s,a) = 1/W (uniform prior).
    AlphaZero: Q + c_puct * P * sqrt(N_parent) / (1 + N_sa)
    """
    if n_parent < 0 or n_sa < 0:
        raise ValueError("visit counts must be non-negative")
    if variant == "paper":
        exploration = nu_expl * prior * math.sqrt(n_parent) / (1.0 + n_sa)
        return q_sa + exploration
    exploration = c_puct * prior * math.sqrt(n_parent) / (1.0 + n_sa)
    return q_sa + exploration


def select_action(
    variant: PUCTVariant,
    nu_expl: float,
    priors: List[float],
    ns: List[int],
    qs: List[float],
    n_parent: int,
) -> int:
    scores = [
        puct_score(variant, nu_expl, priors[a], n_parent, ns[a], float(qs[a]))
        for a in range(len(priors))
    ]
    return int(max(range(len(scores)), key=lambda i: scores[i]))
