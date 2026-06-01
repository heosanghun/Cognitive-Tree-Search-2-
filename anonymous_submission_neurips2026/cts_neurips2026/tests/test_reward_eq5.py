"""Tests for paper Eq.(5) reward function."""

from cts.rewards.shaping import paper_reward, total_reward_stub


def test_paper_reward_correct():
    r = paper_reward(correct=True, terminal_depth=10, lambda_halt=0.05)
    assert r == 1.0 - 0.05 * 10  # 0.5


def test_paper_reward_incorrect():
    r = paper_reward(correct=False, terminal_depth=5, lambda_halt=0.05)
    assert r == 0.0 - 0.05 * 5  # -0.25


def test_paper_reward_zero_depth():
    r = paper_reward(correct=True, terminal_depth=0, lambda_halt=0.05)
    assert r == 1.0


def test_legacy_stub_still_works():
    r = total_reward_stub(0.0, True, 100.0, 0.01)
    assert r == 1.0 - 0.01 * 100.0
