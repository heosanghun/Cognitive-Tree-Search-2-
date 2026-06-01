import torch

from cts.policy.meta_policy import MetaPolicy
from cts.train.stage2_ppo import run_mini_ppo_step


def test_mini_ppo_step():
    m = MetaPolicy(text_dim=64, W=3)
    obs = torch.randn(64)
    loss, _ = run_mini_ppo_step(m, obs=obs, old_action=1, advantage=0.5)
    assert isinstance(loss, float)
