import torch

from cts.policy.meta_policy import MetaPolicy


def test_logits_and_nu_matches_forward_priors():
    m = MetaPolicy(text_dim=16, hidden=8, W=3)
    x = torch.randn(16)
    nu1, logits = m.logits_and_nu(x)
    nu2, p = m.forward(x)
    assert nu1.nu_temp == nu2.nu_temp
    logits2 = m.head_prior(m.act(m.enc(x.unsqueeze(0)))).squeeze(0)
    assert torch.allclose(logits, logits2)
    import math

    assert math.isclose(sum(p), 1.0, rel_tol=1e-5)
