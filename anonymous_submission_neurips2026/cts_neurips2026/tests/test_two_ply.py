from cts.mcts.deep_rollout import two_ply_mcts_rollouts


def test_two_ply_runs():
    out = two_ply_mcts_rollouts("q", sims_root=2, sims_child=2, W=3, d=16)
    assert out.best_action in (0, 1, 2)
    assert len(out.root.transitions) == 2
    assert len(out.child.transitions) == 2
