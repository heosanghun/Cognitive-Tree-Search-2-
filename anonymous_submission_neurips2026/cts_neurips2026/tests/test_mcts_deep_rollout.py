"""N-ply MCTS anchor chain (extends 2-ply)."""

from cts.mcts.mcts_deep_rollout import multi_ply_mcts_rollouts


def test_multi_ply_n1_structure_matches_two_ply_depth():
    m = multi_ply_mcts_rollouts("Q: 1+1?", n_plies=1, sims_per_ply=2, W=3, K=2, d=16)
    assert len(m.anchors) == 2
    assert len(m.transitions) == 1
    assert m.anchors[0] == "Q: 1+1?"
    assert m.anchors[1].startswith("Q: 1+1?")
    assert len(m.rollouts_per_ply) == 2  # root + leaf


def test_multi_ply_n2_has_three_anchors():
    m = multi_ply_mcts_rollouts("root", n_plies=2, sims_per_ply=2, W=3, K=2, d=16)
    assert len(m.anchors) == 3
    assert len(m.transitions) == 2
    assert isinstance(m.leaf_mean_q, float)
