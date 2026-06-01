from cts.mcts.episode import default_transition_reward, mcts_root_rollouts, puct_select_and_expand_once
from cts.types import RuntimeBudgetState, TransitionResult


def test_rollouts_sum_ns_equals_sims():
    out = mcts_root_rollouts("q", num_simulations=5, W=3, d=16)
    assert sum(out.ns) == 5
    assert len(out.transitions) == 5


def test_custom_reward_fn():
    def always_half(_r):
        return 0.5

    out = mcts_root_rollouts("x", num_simulations=2, W=3, d=16, reward_fn=always_half)
    assert all(0.0 <= q <= 1.0 for q in out.qs)


def test_default_reward():
    r = TransitionResult(
        child_text="x",
        z_star_child=__import__("torch").zeros(2, 2),
        solver_stats={"converged": True},
        prune=False,
        budget=RuntimeBudgetState(),
    )
    assert default_transition_reward(r) == 1.0


def test_puct_once_accepts_reward_fn():
    out = puct_select_and_expand_once(
        "y",
        W=3,
        d=16,
        reward_fn=lambda _r: 0.25,
    )
    assert out.transition is not None
