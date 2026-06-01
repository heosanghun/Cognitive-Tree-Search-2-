"""Tests for NeuroCritic batch evaluation (paper §5.3)."""

import torch

from cts.critic.neuro_critic import NeuroCritic


def test_critic_single():
    critic = NeuroCritic(z_dim=32)
    z = torch.randn(32)
    v = critic(z)
    assert v.shape == (1, 1)


def test_critic_batch():
    critic = NeuroCritic(z_dim=32)
    z = torch.randn(4, 32)
    v = critic(z)
    assert v.shape == (4, 1)


def test_critic_batch_evaluate():
    critic = NeuroCritic(z_dim=32)
    z_star_batch = torch.randn(3, 8, 32)  # [W, K, d]
    v = critic.batch_evaluate(z_star_batch)
    assert v.shape == (3, 1)


def test_critic_batch_evaluate_different_d():
    critic = NeuroCritic(z_dim=64)
    z_star_batch = torch.randn(3, 8, 32)  # d < z_dim
    v = critic.batch_evaluate(z_star_batch)
    assert v.shape == (3, 1)
