from cts.train.ppo_core import compute_gae


def test_gae_shape():
    r = [1.0, 0.0]
    v = [0.5, 0.5]
    d = [False, True]
    adv, rets = compute_gae(r, v, d, gamma=0.9, lam=0.9)
    assert len(adv) == 2
    assert len(rets) == 2
