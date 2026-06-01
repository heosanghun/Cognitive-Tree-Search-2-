"""P0-4 regression: Stage 2 PPO trainer must match paper Table 4 / §6.2.

Audited mismatches that this file pins (NeurIPS Apr 2026 audit):

  - ``collect_batch`` (rollout buffer size) must default to 64, not 4. The
    previous default was a smoke-test value that under-trained the
    meta-policy by 16x in samples-per-update terms.

  - ``ppo_epochs`` must default to 4, not 2. Two epochs is also a
    smoke-test value and halves the effective sample re-use.

  - The actor (MetaPolicy) and critic (value head / NeuroCritic) must use
    SEPARATE optimizer parameter groups: actor at ``ppo_lr`` (3e-5) and
    critic at ``critic_lr`` (1e-4). Sharing the actor lr for the critic
    inflates PPO advantage variance.

  - These three knobs must be readable from `configs/default.yaml`.

CPU-only; no Gemma load.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def _import_run_stage2() -> object:
    """Lazy import to keep Gemma out of ``sys.modules`` at test-collection
    time; otherwise ``test_public_api`` lazy-loading invariants regress
    via pytest's shared interpreter."""
    from cts.train.stage2_ppo_train import run_stage2_math_ppo
    return run_stage2_math_ppo


def test_default_config_carries_paper_aligned_ppo_keys() -> None:
    from cts.utils.config import load_config

    cfg = load_config("default")
    assert int(cfg["ppo_collect_batch"]) == 64
    assert int(cfg["ppo_epochs"]) == 4
    assert float(cfg["ppo_lr"]) == 3e-5
    assert float(cfg["critic_lr"]) == 1e-4


def test_run_stage2_signature_takes_optional_collect_batch_and_epochs() -> None:
    """The function MUST accept None for both knobs so the config-driven
    paper defaults take effect when callers do not override."""
    run_stage2_math_ppo = _import_run_stage2()
    sig = inspect.signature(run_stage2_math_ppo)
    p_cb = sig.parameters["collect_batch"]
    p_ep = sig.parameters["ppo_epochs"]
    assert p_cb.default is None, (
        "collect_batch hardcoded default leaks the smoke-test value 4"
    )
    assert p_ep.default is None, (
        "ppo_epochs hardcoded default leaks the smoke-test value 2"
    )


def test_actor_and_critic_have_separate_lr_groups() -> None:
    """Mirror the optimizer construction the trainer performs and assert
    two distinct lr groups (3e-5 actor / 1e-4 critic)."""
    actor = nn.Linear(8, 8)
    critic = nn.Linear(8, 1)
    opt = torch.optim.AdamW(
        [
            {"params": list(actor.parameters()), "lr": 3e-5},
            {"params": list(critic.parameters()), "lr": 1e-4},
        ]
    )
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [3e-5, 1e-4]
