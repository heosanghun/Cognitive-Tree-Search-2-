"""CPU-only regression tests for P1 baseline modules (no Gemma load)."""

from __future__ import annotations

import torch

from cts.baselines.bon_critic import bon_select_pred_with_critic
from cts.baselines.ucb1_nu import UCB1NuExplBandit


def test_ucb1_explores_unplayed_arms_first():
    b = UCB1NuExplBandit(n_arms=5)
    arms = []
    for _ in range(5):
        arm, _ = b.select()
        arms.append(arm)
        b.update(arm, 0.0)
    assert len(set(arms)) == 5


def test_ucb1_update_and_prefers_high_reward_arm():
    b = UCB1NuExplBandit(n_arms=3)
    for _ in range(3):
        b.select()
        b.update(0, 0.0)
    # Warm all arms once
    for arm in range(3):
        b.select()
        b.update(arm, 1.0 if arm == 2 else 0.0)
    # After many pulls arm 2 should win often
    picks = [b.select()[0] for _ in range(20)]
    assert picks.count(2) >= 10


def test_bon_select_with_mock_critic():
    class _BB:
        def encode_context(self, text: str) -> torch.Tensor:
            # Higher score for longer text in this toy setup
            return torch.tensor([float(len(text))])

    class _Critic(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.sum(dim=-1, keepdim=True)

    pred = bon_select_pred_with_critic(
        critic=_Critic(),
        backbone=_BB(),
        raw_candidates=["short", "much longer candidate chain"],
        extract_pred_fn=lambda raw, bench: raw[:8],
        benchmark="math500",
        device=torch.device("cpu"),
    )
    assert pred == "much lon"


def test_run_cts_eval_full_mentions_ft_nt_lora_loader():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_cts_eval_full.py"
    ).read_text(encoding="utf-8")
    assert "build_ft_nt_predictor" in src
    assert "UCB1NuExplBandit" in src
    assert "bon_select_pred_with_critic" in src
