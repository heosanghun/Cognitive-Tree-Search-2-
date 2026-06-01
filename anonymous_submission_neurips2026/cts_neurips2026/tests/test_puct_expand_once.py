import torch

from cts.mcts.episode import puct_select_and_expand_once
from cts.policy.meta_policy import MetaPolicy
from cts.types import NuVector


def test_puct_expand_once_without_meta():
    out = puct_select_and_expand_once("x", W=3, K=4, d=16)
    assert 0 <= out.selected_action < 3
    assert isinstance(out.nu, NuVector)
    assert len(out.priors) == 3


def test_puct_expand_once_with_meta_policy():
    mp = MetaPolicy(text_dim=64, W=3)
    out = puct_select_and_expand_once(
        "hello",
        W=3,
        d=32,
        meta_policy=mp,
    )
    assert len(out.priors) == 3
    assert out.tree.nodes[0].mcts_prior == out.priors


def test_custom_text_features_override():
    mp = MetaPolicy(text_dim=64, W=3)
    feats = torch.randn(64)
    out = puct_select_and_expand_once(
        "ignored",
        meta_policy=mp,
        text_features=feats,
    )
    assert out.selected_action in (0, 1, 2)
