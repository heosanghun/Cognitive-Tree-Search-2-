import torch

from cts.critic.neuro_critic import NeuroCritic
from cts.mcts.critic_reward import make_critic_reward_fn, z_star_to_vector
from cts.types import RuntimeBudgetState, TransitionResult


def test_z_star_to_vector_pad():
    z = torch.randn(4, 8)
    v = z_star_to_vector(z, 16)
    assert v.shape == (16,)


def test_critic_reward_fn():
    c = NeuroCritic(8)
    fn = make_critic_reward_fn(c, z_dim=8)
    r = TransitionResult(
        child_text="x",
        z_star_child=torch.randn(3, 8),
        solver_stats={},
        prune=False,
        budget=RuntimeBudgetState(),
    )
    x = fn(r)
    assert 0.0 <= x <= 1.0
