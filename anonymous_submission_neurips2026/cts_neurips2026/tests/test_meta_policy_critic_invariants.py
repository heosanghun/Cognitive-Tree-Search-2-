"""Reviewer-grade invariant tests for ``cts.policy.meta_policy.MetaPolicy``
and ``cts.critic.neuro_critic.NeuroCritic`` (paper &sect;4.1, &sect;5.3).

These tests pin down properties that are easy to break in a refactor but
that the paper's algorithmic correctness depends on:

* ``MetaPolicy.logits_and_nu`` must yield a well-formed ``NuVector``
  whose components stay in the documented ranges.
* Branch priors must form a probability distribution after softmax.
* Backward-pass through the policy and critic must produce gradients on
  every parameter (otherwise PPO updates silently no-op).
* Different inputs must give different outputs (no degenerate constant
  predictions).
* ``NeuroCritic.batch_evaluate`` must return the expected shape on
  paper-realistic ``[W, K, d]`` tensors.
"""

from __future__ import annotations

import torch

from cts.policy.meta_policy import MetaPolicy
from cts.critic.neuro_critic import NeuroCritic
from cts.types import NuVector


# ---------------------------------------------------------------------------
# MetaPolicy
# ---------------------------------------------------------------------------


def test_meta_policy_nu_components_satisfy_paper_ranges():
    torch.manual_seed(0)
    m = MetaPolicy(text_dim=16, hidden=32, W=3)
    for _ in range(50):
        x = torch.randn(16) * 3.0  # try a wide range of inputs
        nu, _ = m.logits_and_nu(x)
        assert isinstance(nu, NuVector)
        # softplus(.) + 0.5 => >= 0.5
        assert nu.nu_expl >= 0.5
        assert nu.nu_temp >= 0.5
        assert nu.nu_act >= 0.5
        # sigmoid(.) => in (0, 1)
        assert 0.0 < nu.nu_tol < 1.0


def test_meta_policy_branch_priors_are_a_probability_distribution():
    torch.manual_seed(1)
    m = MetaPolicy(text_dim=16, hidden=8, W=3)
    x = torch.randn(16)
    _, priors = m(x)
    assert len(priors) == 3
    assert all(0.0 <= p <= 1.0 for p in priors)
    assert abs(sum(priors) - 1.0) < 1e-5


def test_meta_policy_branch_logits_have_grad_through_loss():
    torch.manual_seed(2)
    m = MetaPolicy(text_dim=16, hidden=8, W=3)
    x = torch.randn(16)
    _, logits = m.logits_and_nu(x)
    target = torch.tensor(1)  # arbitrary target branch
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    # head_prior must have a gradient
    assert m.head_prior.weight.grad is not None
    assert m.head_prior.weight.grad.abs().sum().item() > 0
    # encoder must also receive gradient (shared backbone)
    assert m.enc.weight.grad is not None
    assert m.enc.weight.grad.abs().sum().item() > 0


def test_meta_policy_different_inputs_give_different_priors():
    torch.manual_seed(3)
    m = MetaPolicy(text_dim=16, hidden=16, W=3)
    a = torch.zeros(16)
    b = torch.ones(16)
    _, pa = m(a)
    _, pb = m(b)
    # Outputs must not collapse to identical priors for different inputs
    assert any(abs(pa[i] - pb[i]) > 1e-4 for i in range(3))


def test_meta_policy_accepts_batched_input_unsqueezed():
    """Both [text_dim] and [1, text_dim] inputs must work (caller-friendly)."""
    torch.manual_seed(4)
    m = MetaPolicy(text_dim=16, hidden=8, W=3)
    x_unbatched = torch.randn(16)
    x_batched = x_unbatched.unsqueeze(0)
    nu1, l1 = m.logits_and_nu(x_unbatched)
    nu2, l2 = m.logits_and_nu(x_batched)
    assert nu1.nu_expl == nu2.nu_expl
    assert torch.allclose(l1, l2)


# ---------------------------------------------------------------------------
# NeuroCritic
# ---------------------------------------------------------------------------


def test_critic_value_grad_flows_through_mse_loss():
    torch.manual_seed(5)
    critic = NeuroCritic(z_dim=32)
    z = torch.randn(4, 32)
    v = critic(z)
    target = torch.zeros_like(v)
    loss = torch.nn.functional.mse_loss(v, target)
    loss.backward()
    for name, p in critic.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert p.grad.abs().sum().item() > 0, f"zero grad on {name}"


def test_critic_value_changes_when_input_changes():
    torch.manual_seed(6)
    critic = NeuroCritic(z_dim=32)
    a = torch.zeros(1, 32)
    b = torch.ones(1, 32)
    va = critic(a)
    vb = critic(b)
    assert (va - vb).abs().item() > 1e-4


def test_critic_batch_evaluate_handles_paper_realistic_shape():
    """Paper §4.1: W=3 branches, each with K=64 latent tokens, d=z_dim."""
    critic = NeuroCritic(z_dim=128)
    z_star_batch = torch.randn(3, 64, 128)
    v = critic.batch_evaluate(z_star_batch)
    assert v.shape == (3, 1)
    assert torch.isfinite(v).all()


def test_critic_batch_evaluate_pads_when_d_smaller_than_z_dim():
    """d < z_dim path must zero-pad rather than crash."""
    critic = NeuroCritic(z_dim=128)
    z_star_batch = torch.randn(3, 64, 32)  # d=32 < z_dim=128
    v = critic.batch_evaluate(z_star_batch)
    assert v.shape == (3, 1)


def test_critic_batch_evaluate_truncates_when_d_larger_than_z_dim():
    """d > z_dim path must truncate rather than crash."""
    critic = NeuroCritic(z_dim=32)
    z_star_batch = torch.randn(3, 64, 128)  # d=128 > z_dim=32
    v = critic.batch_evaluate(z_star_batch)
    assert v.shape == (3, 1)
