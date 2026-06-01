"""Numerical-correctness tests for ``cts.train.ppo_core``.

These tests pin down the *exact* computation done by ``compute_gae`` and
``ppo_clipped_loss`` against hand-derived reference values. The pre-existing
``tests/test_gae.py`` only checks shape; reviewer-grade audit needs
specific numerics for GAE bootstrapping, the clipping boundary, and the
sign of the gradient. (Paper Table 4 + &sect;6.2 PPO.)
"""

from __future__ import annotations

import math

import pytest
import torch

from cts.train.ppo_core import compute_gae, ppo_clipped_loss, value_loss


# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------


def test_gae_terminal_step_truncates_bootstrap():
    """When dones[t]=True, V[t+1] is masked out per Schulman et al. (2016)."""
    rewards = [1.0]
    values = [0.0]
    dones = [True]
    adv, rets = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv == pytest.approx([1.0])
    assert rets == pytest.approx([1.0])


def test_gae_nonterminal_uses_next_value():
    """delta_0 = r_0 + gamma * V_1 - V_0 when dones[0]=False."""
    rewards = [1.0, 2.0]
    values = [0.5, 1.0]
    dones = [False, True]
    gamma, lam = 0.9, 0.5
    adv, rets = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    delta_1 = 2.0 + gamma * 0.0 - 1.0
    delta_0 = 1.0 + gamma * 1.0 - 0.5
    expected_adv1 = delta_1
    expected_adv0 = delta_0 + gamma * lam * delta_1
    assert adv == pytest.approx([expected_adv0, expected_adv1])
    assert rets == pytest.approx([expected_adv0 + 0.5, expected_adv1 + 1.0])


def test_gae_three_step_against_hand_calc():
    """Closed-form 3-step horizon to detect off-by-one in the recursion."""
    rewards = [1.0, 0.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [False, False, True]
    gamma, lam = 0.9, 0.95
    adv, rets = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    delta_2 = 1.0 - 0.5
    delta_1 = 0.0 + gamma * 0.5 - 0.5
    delta_0 = 1.0 + gamma * 0.5 - 0.5
    a2 = delta_2
    a1 = delta_1 + gamma * lam * a2
    a0 = delta_0 + gamma * lam * a1
    assert adv == pytest.approx([a0, a1, a2])


def test_gae_paper_table4_gamma_lambda_defaults_match_function_signature():
    """Paper Table 4: discount gamma = 0.99, GAE lambda = 0.95.

    The function defaults must match so a future call site that omits the
    keyword arguments does not silently use a different discount.
    """
    import inspect
    sig = inspect.signature(compute_gae)
    assert sig.parameters["gamma"].default == 0.99
    assert sig.parameters["lam"].default == 0.95


def test_gae_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError):
        compute_gae([1.0, 2.0], [0.5], [False, False])
    with pytest.raises(ValueError):
        compute_gae([1.0], [0.5], [False, False])


def test_gae_zero_rewards_zero_values_yields_zero_advantage():
    """Sanity: no signal, no advantage."""
    rewards = [0.0] * 5
    values = [0.0] * 5
    dones = [False] * 4 + [True]
    adv, rets = compute_gae(rewards, values, dones)
    assert adv == pytest.approx([0.0] * 5)
    assert rets == pytest.approx([0.0] * 5)


def test_gae_monotonic_in_rewards():
    """Larger reward at step 0 => larger advantage at step 0."""
    base, _ = compute_gae([0.5, 0.0], [0.0, 0.0], [False, True])
    bigger, _ = compute_gae([1.5, 0.0], [0.0, 0.0], [False, True])
    assert bigger[0] > base[0]


# ---------------------------------------------------------------------------
# ppo_clipped_loss
# ---------------------------------------------------------------------------


def test_ppo_loss_zero_when_ratio_is_one_and_advantage_is_zero():
    new_logp = torch.tensor([0.0, 0.0])
    old_logp = torch.tensor([0.0, 0.0])
    adv = torch.tensor([0.0, 0.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv)
    assert loss.item() == pytest.approx(0.0)


def test_ppo_loss_negative_advantage_zero_logp_diff_equals_zero():
    """ratio=1, so loss = -mean(adv); sign convention check."""
    new_logp = torch.tensor([0.0])
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv)
    # ratio=1; min(1*1, 1*1) = 1; -mean = -1
    assert loss.item() == pytest.approx(-1.0)


def test_ppo_loss_clip_upper_boundary_positive_advantage():
    """When new_logp - old_logp > log(1+clip), the clipped term wins.

    With clip=0.2 and log_ratio=ln(2.0) (=> ratio=2), advantage > 0:
        unclipped = 2.0 * adv
        clipped   = 1.2 * adv
        min(...)  = 1.2 * adv  (clipped is smaller, gets selected)
        loss      = -1.2 * adv
    """
    new_logp = torch.tensor([math.log(2.0)])
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip=0.2)
    assert loss.item() == pytest.approx(-1.2, abs=1e-6)


def test_ppo_loss_clip_lower_boundary_negative_advantage():
    """When new_logp - old_logp < log(1-clip) and adv < 0, clipped term wins.

    With clip=0.2 and ratio=0.5, adv = -1.0:
        unclipped = 0.5 * -1.0 = -0.5
        clipped   = 0.8 * -1.0 = -0.8
        min(...)  = -0.8  (clipped is smaller, gets selected)
        loss      = -mean(-0.8) = 0.8
    """
    new_logp = torch.tensor([math.log(0.5)])
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([-1.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip=0.2)
    assert loss.item() == pytest.approx(0.8, abs=1e-6)


def test_ppo_loss_within_clip_band_uses_unclipped_term():
    """Inside the trust region, the loss reduces to plain policy gradient."""
    new_logp = torch.tensor([0.05])  # ratio ~= 1.05, inside [0.8, 1.2]
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip=0.2)
    expected_ratio = math.exp(0.05)
    assert loss.item() == pytest.approx(-expected_ratio, rel=1e-5)


def test_ppo_loss_gradient_pushes_logp_toward_higher_adv():
    """Sign-of-gradient test: positive advantage => increasing new_logp lowers loss."""
    new_logp = torch.tensor([0.0], requires_grad=True)
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip=0.2)
    loss.backward()
    assert new_logp.grad is not None
    # d(-ratio * adv)/d(new_logp) = -ratio*adv = -1*1 = -1 (inside band)
    assert new_logp.grad.item() == pytest.approx(-1.0, abs=1e-6)


def test_ppo_loss_gradient_pushes_logp_away_from_negative_adv():
    new_logp = torch.tensor([0.0], requires_grad=True)
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([-1.0])
    loss = ppo_clipped_loss(new_logp, old_logp, adv, clip=0.2)
    loss.backward()
    # d(-ratio * adv)/d(new_logp) = -1 * -1 = 1; positive => decrease new_logp
    assert new_logp.grad.item() == pytest.approx(1.0, abs=1e-6)


def test_ppo_loss_default_clip_matches_paper_table4():
    import inspect
    sig = inspect.signature(ppo_clipped_loss)
    assert sig.parameters["clip"].default == 0.2


# ---------------------------------------------------------------------------
# value_loss
# ---------------------------------------------------------------------------


def test_value_loss_matches_torch_mse():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 2.0, 5.0])
    loss = value_loss(pred, target)
    expected = ((1.0) ** 2 + 0.0 + (2.0) ** 2) / 3
    assert loss.item() == pytest.approx(expected)


def test_value_loss_zero_when_pred_equals_target():
    pred = torch.tensor([1.0, 2.0, 3.0])
    loss = value_loss(pred, pred.clone())
    assert loss.item() == pytest.approx(0.0)


def test_value_loss_gradient_points_pred_toward_target():
    pred = torch.tensor([0.0], requires_grad=True)
    target = torch.tensor([1.0])
    loss = value_loss(pred, target)
    loss.backward()
    # d/d_pred 0.5*(pred-target)^2 with mean-reduce on size-1 = (pred-target) = -1
    assert pred.grad is not None
    assert pred.grad.item() == pytest.approx(-2.0, abs=1e-6)
