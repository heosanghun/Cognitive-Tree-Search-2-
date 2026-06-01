"""Reward shaping: paper Eq.(5) — R_total = 1{correct} − λ_halt · T."""

from __future__ import annotations


def paper_reward(
    correct: bool,
    terminal_depth: int,
    lambda_halt: float = 0.05,
) -> float:
    """Paper Eq.(5): binary outcome minus compute penalty.

    Args:
        correct: whether the final answer matches ground truth
        terminal_depth: T — actual accumulated terminal tree depth (or MACs consumed)
        lambda_halt: λ_halt penalty coefficient (paper Table 4: 0.05)

    Returns:
        R_total = 1{correct_answer} − λ_halt · T
    """
    indicator = 1.0 if correct else 0.0
    return indicator - lambda_halt * terminal_depth


def total_reward_stub(
    process_term: float, correct: bool, ado_accumulated: float, lam: float
) -> float:
    """Legacy reward stub — prefer paper_reward() for paper alignment."""
    term = 1.0 if correct else 0.0
    return process_term + term - lam * ado_accumulated
