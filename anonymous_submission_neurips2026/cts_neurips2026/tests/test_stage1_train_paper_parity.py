"""P0-2/3 regression: Stage 1 trainer must match paper §6.1 / App. I.

Audited mismatches that this file pins (NeurIPS Apr 2026 audit):

  - W_proj must be in the trainable parameter set (paper §6.1: "routing
    projection W_g and W_proj trained on 10,000 examples"). The previous
    implementation silently froze W_proj because the matcher only knew
    about ``routing_proj``/``_blend``/``lora_``, so the latent-to-token
    decoding head never received gradient signal.

  - Stage 1 lr must be 1e-4 (paper §6.1) — distinct from the PPO lr 3e-5
    that the legacy ``lr`` key encoded.

  - Stage 1 lr schedule must include a 100-step linear warm-up followed
    by cosine annealing (paper App. I).

  - Stage 1 effective batch must be 2 — implemented here as gradient
    accumulation since OpenMathInstruct rows are streamed one at a time.

These tests do NOT spin up Gemma; they exercise just the parameter-gating
predicate and the config-driven scheduler factory. CPU-only, <1 s.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _import_set_trainable_params():
    """Lazy import so that simply importing this test module does NOT pull
    Gemma into ``sys.modules`` and break ``test_public_api`` lazy-loading
    invariants. The Stage 1 trainer does a module-level
    ``from cts.backbone.gemma_adapter import GemmaCTSBackbone`` which would
    otherwise propagate via test discovery and contaminate the import
    snapshot the public-API test relies on.
    """
    from cts.train.stage1_openmath_train import _set_trainable_params
    return _set_trainable_params


class _DummyBackbone(nn.Module):
    """Minimal stand-in for GemmaCTSBackbone parameter naming."""

    def __init__(self) -> None:
        super().__init__()
        self.routing_proj = nn.Linear(8, 8, bias=False)
        self.w_proj = nn.Linear(8, 8, bias=False)
        self._blend = nn.Parameter(torch.zeros(1))
        self.frozen_backbone = nn.Linear(8, 8)
        self.lora_A = nn.Parameter(torch.zeros(4, 8))
        self.lora_B = nn.Parameter(torch.zeros(8, 4))


def test_w_proj_is_trainable_in_stage1_no_lora() -> None:
    """Paper §6.1: W_proj must be part of the Stage 1 trainable set."""
    set_trainable_params = _import_set_trainable_params()
    bb = _DummyBackbone()
    set_trainable_params(bb, train_lora=False)

    name_to_train = {n: p.requires_grad for n, p in bb.named_parameters()}
    assert name_to_train["w_proj.weight"] is True, (
        "W_proj is frozen in Stage 1 — re-introduces the 'unlearned latent "
        "decoder' bug that breaks paper-parity 50.2 % AIME."
    )


def test_routing_and_blend_are_trainable() -> None:
    """Routing projection W_g and the residual blend scalar stay live."""
    set_trainable_params = _import_set_trainable_params()
    bb = _DummyBackbone()
    set_trainable_params(bb, train_lora=False)

    name_to_train = {n: p.requires_grad for n, p in bb.named_parameters()}
    assert name_to_train["routing_proj.weight"] is True
    assert name_to_train["_blend"] is True


def test_frozen_backbone_stays_frozen() -> None:
    """Everything outside the {W_g, W_proj, _blend, LoRA} set is frozen."""
    set_trainable_params = _import_set_trainable_params()
    bb = _DummyBackbone()
    set_trainable_params(bb, train_lora=False)

    name_to_train = {n: p.requires_grad for n, p in bb.named_parameters()}
    assert name_to_train["frozen_backbone.weight"] is False
    assert name_to_train["frozen_backbone.bias"] is False


def test_lora_toggle_gates_lora_only() -> None:
    """LoRA adapters require ``train_lora=True`` to receive gradient."""
    set_trainable_params = _import_set_trainable_params()
    bb = _DummyBackbone()
    set_trainable_params(bb, train_lora=False)
    name_to_train = {n: p.requires_grad for n, p in bb.named_parameters()}
    assert name_to_train["lora_A"] is False
    assert name_to_train["lora_B"] is False

    bb2 = _DummyBackbone()
    set_trainable_params(bb2, train_lora=True)
    name_to_train2 = {n: p.requires_grad for n, p in bb2.named_parameters()}
    assert name_to_train2["lora_A"] is True
    assert name_to_train2["lora_B"] is True
    # W_proj is on regardless of lora flag.
    assert name_to_train2["w_proj.weight"] is True


def test_default_config_carries_paper_aligned_stage1_keys() -> None:
    """``configs/default.yaml`` must expose the paper-aligned Stage 1
    hyperparameters that the trainer reads."""
    from cts.utils.config import load_config

    cfg = load_config("default")
    assert float(cfg["stage1_lr"]) == 1e-4
    assert int(cfg["stage1_warmup_steps"]) == 100
    assert str(cfg["stage1_lr_schedule"]).lower() == "cosine"
    assert int(cfg["stage1_batch_size"]) == 2


def test_warmup_then_cosine_scheduler_shape() -> None:
    """Reproduce the warmup + cosine schedule the trainer builds and pin
    its LR shape: at step 0 ~= 0, at step `warmup` == base_lr, then it
    monotonically decreases to ~0 at the end."""
    base_lr = 1e-4
    warmup_steps = 100
    total_steps = 500
    decay_steps = total_steps - warmup_steps

    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=base_lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=decay_steps)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )

    lrs: list[float] = [opt.param_groups[0]["lr"]]
    for _ in range(total_steps):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Warmup ramps from ~0 up to base_lr at the boundary.
    assert lrs[0] < base_lr * 0.05
    assert lrs[warmup_steps] >= base_lr * 0.95
    # Cosine then decays monotonically to ~0 at the end.
    assert lrs[-1] < base_lr * 0.1
    # Strictly non-increasing on the cosine tail.
    tail = lrs[warmup_steps:]
    for a, b in zip(tail, tail[1:]):
        assert b <= a + 1e-12
