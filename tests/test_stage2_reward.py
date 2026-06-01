"""CPU-only tests for Stage-2 rollout reward (paper Eq. 5)."""

from __future__ import annotations

from cts.train.stage2_reward import (
    gold_from_stage2_row,
    stage2_rollout_is_correct,
    stage2_rollout_reward,
)


def test_gold_from_row_solution_key():
    row = {"prompt": "x", "solution": "Therefore \\boxed{42}"}
    assert gold_from_stage2_row(row) == "Therefore \\boxed{42}"


def test_gold_from_row_missing_returns_none():
    assert gold_from_stage2_row({"prompt": "only prompt"}) is None


def test_converged_mode_ignores_gold():
    row = {"prompt": "p", "solution": "\\boxed{7}"}
    assert stage2_rollout_is_correct(
        row, "wrong", converged=True, reward_mode="converged"
    )
    assert not stage2_rollout_is_correct(
        row, "wrong", converged=False, reward_mode="converged"
    )


def test_auto_mode_falls_back_to_converged_without_gold():
    row = {"prompt": "p"}
    assert stage2_rollout_is_correct(
        row, "anything", converged=True, reward_mode="auto"
    )
    assert not stage2_rollout_is_correct(
        row, "anything", converged=False, reward_mode="auto"
    )


def test_auto_mode_uses_answer_when_gold_and_child_text_match():
    row = {"prompt": "p", "solution": "Final answer: \\boxed{12}"}
    assert stage2_rollout_is_correct(
        row, "So \\boxed{12}", converged=False, reward_mode="auto"
    )


def test_answer_mode_false_without_gold():
    row = {"prompt": "p"}
    assert not stage2_rollout_is_correct(
        row, "\\boxed{1}", converged=True, reward_mode="answer"
    )


def test_stage2_rollout_reward_eq5_shape():
    r = stage2_rollout_reward(
        {"prompt": "p"},
        child_text="",
        converged=True,
        terminal_depth=10,
        lambda_halt=0.05,
        reward_mode="auto",
    )
    assert r == 0.5  # 1.0 - 0.05 * 10
