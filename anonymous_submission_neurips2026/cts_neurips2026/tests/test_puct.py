import math

from cts.mcts.puct import puct_score, select_action


def test_puct_paper_exploration_term():
    nu, p, npa, nsa = 2.0, 0.3, 10, 2
    q = 0.1
    u = puct_score("paper", nu, p, npa, nsa, q)
    expected = q + nu * p * math.sqrt(npa) / (1.0 + nsa)
    assert abs(u - expected) < 1e-6


def test_select_action_prefers_high_prior():
    priors = [0.5, 0.25, 0.25]
    ns = [0, 0, 0]
    qs = [0.0, 0.0, 0.0]
    a = select_action("paper", 1.0, priors, ns, qs, n_parent=1)
    assert a == 0
