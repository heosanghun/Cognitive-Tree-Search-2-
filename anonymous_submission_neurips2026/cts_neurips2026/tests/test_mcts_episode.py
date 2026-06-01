from cts.mcts.episode import expand_root_parallel_branches


def test_expand_root_creates_w_children():
    tree, results = expand_root_parallel_branches("x", W=3, K=4, d=16)
    assert len(tree.nodes) == 1 + 3
    assert len(results) == 3
    assert len(tree.root().children_ids) == 3
