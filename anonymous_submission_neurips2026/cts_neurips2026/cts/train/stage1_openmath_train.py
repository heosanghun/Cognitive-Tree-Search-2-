"""
Stage 1: OpenMathInstruct JSONL + GemmaCTSBackbone + IFT residual loss (paper §6.1).

Base Gemma weights are frozen; trains `routing_proj`, `_blend`, and optional LoRA adapters.
LoRA r=8, ~18 MB trainable parameters. 10,000 examples from OpenMathInstruct-2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.jsonl_iter import iter_jsonl
from cts.train.lora_compat import apply_paper_lora
from cts.train.openmath_text import prompt_text_from_openmath_row
from cts.train.stage1_warmup import fixed_point_surrogate_loss
from cts.types import NuVector
from cts.utils.config import load_config
from cts.utils.repro_seed import apply_global_seed


def _set_trainable_params(bb: GemmaCTSBackbone, *, train_lora: bool) -> None:
    """Enable gradients for the Stage 1 trainable parameter set (paper §6.1).

    Paper §6.1: "Frozen Gemma 4; LoRA (r=8, alpha=16) at q/v/o_proj; routing
    projection W_g; and W_proj trained on 10,000 OpenMathInstruct-2 examples."

    Trainable set therefore = ``routing_proj`` (W_g), ``w_proj`` (W_proj),
    LoRA adapters (when ``train_lora=True``), plus the residual ``_blend``
    convex-combination scalar that the GemmaCTSBackbone exposes for
    Wproj fusion. Everything else is held frozen.
    """
    for n, p in bb.named_parameters():
        if "routing_proj" in n or "_blend" in n:
            p.requires_grad = True
        elif "w_proj" in n:
            # Paper-aligned: W_proj is part of the Stage 1 learned set.
            # Previously this parameter was silently frozen (NeurIPS audit
            # Apr 2026 finding P0-2), so the latent->token decoding head
            # was never updated and the headline 50.2 % AIME number was
            # not reproducible from these checkpoints.
            p.requires_grad = True
        elif "lora_" in n and train_lora:
            p.requires_grad = True
        else:
            p.requires_grad = False


def _maybe_apply_lora(
    bb: GemmaCTSBackbone,
    *,
    rank: int,
    target_modules: list[str],
) -> GemmaCTSBackbone:
    """Paper §6.1: LoRA r=8, α=16 on q/v/o_proj of the language model.

    Thin wrapper around :func:`cts.train.lora_compat.apply_paper_lora`
    that keeps Stage 1's existing call site stable while moving the
    actual implementation into :mod:`cts.train.lora_compat` so Stage 2
    PPO and the eval pipelines can reuse the same adapter geometry.
    """
    return apply_paper_lora(
        bb,
        rank=rank,
        target_modules=tuple(target_modules),
        dropout=0.05,
        require_match=True,
    )


def _save_checkpoint(
    bb: GemmaCTSBackbone,
    opt: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    config_name: str,
    openmath_jsonl: str,
    lora: bool,
    losses: list[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state_dict": bb.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": step,
            "total_steps": total_steps,
            "config_name": config_name,
            "openmath_jsonl": openmath_jsonl,
            "lora": lora,
            "losses": losses[-200:],
        },
        path,
    )
    print(f"  [checkpoint] saved step {step} → {path}")


def run_stage1_openmath_training(
    *,
    openmath_jsonl: Path | str,
    config_name: str = "default",
    max_steps: Optional[int] = None,
    device: Optional[str] = None,
    lora: bool = False,
    lora_targets: Optional[list[str]] = None,
    log_every: int = 20,
    model_dir: Optional[str] = None,
    resume: bool = False,
    save_every: int = 500,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = load_config(config_name)
    apply_global_seed()
    deq_from_cfg = cfg.get("cts_deq_map_mode")
    if deq_from_cfg and not os.environ.get("CTS_DEQ_MAP_MODE"):
        os.environ["CTS_DEQ_MAP_MODE"] = str(deq_from_cfg)
    steps = int(
        max_steps if max_steps is not None else cfg.get("stage1_max_steps", 5000)
    )
    K = int(cfg.get("soft_thought_K", 64))
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)

    dev_s = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_s)
    map_gpu = dev_s if dev_s.startswith("cuda") else None
    if map_gpu is None and dev.type == "cuda":
        map_gpu = str(dev)

    mid = model_dir or os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=map_gpu if map_gpu else "auto",
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)
    bb.train()

    targets = lora_targets or list(cfg.get("lora_target", ["q_proj", "v_proj"]))
    if lora:
        bb = _maybe_apply_lora(
            bb, rank=int(cfg.get("lora_rank", 8)), target_modules=targets
        )
    _set_trainable_params(bb, train_lora=lora)

    params = [p for p in bb.parameters() if p.requires_grad]
    # Stage 1 lr: prefer the dedicated `stage1_lr` key (paper §6.1: 1e-4)
    # over the legacy `lr` key (which is shared with PPO and defaults to
    # 3e-5). Falling back to `lr` keeps backward compatibility for older
    # configs that did not split the two stages.
    lr = float(cfg.get("stage1_lr", cfg.get("lr", 1e-4)))
    opt = torch.optim.AdamW(params, lr=lr)
    # NOTE: torch.amp.GradScaler is implemented for fp16 only. With bf16
    # autocast (Gemma 4 default; paper §5: bf16 weights), the gradient
    # tensors are bf16 and `scaler.unscale_()` aborts with
    # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
    # 'BFloat16'". We therefore skip the scaler entirely; bf16's wider
    # exponent range makes underflow nearly impossible at the scales we
    # train at, so dynamic loss scaling is unnecessary.
    scaler = None

    # Paper App. I: 100-step linear warm-up + cosine decay over the
    # remaining steps. We compose two SequentialLR-style schedulers.
    warmup_steps = int(cfg.get("stage1_warmup_steps", 100))
    lr_schedule = str(cfg.get("stage1_lr_schedule", "cosine")).lower()
    decay_steps = max(1, steps - warmup_steps)
    if lr_schedule == "cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_steps)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=decay_steps)
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = (
            torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[max(1, warmup_steps)]
            )
        )
    else:
        scheduler = None

    # Paper §6.1: batch size 2. Implemented as gradient accumulation since
    # OpenMathInstruct-2 is iterated row-by-row (each row is one example).
    grad_accum_steps = max(1, int(cfg.get("stage1_batch_size", 1)))

    start_step = 0
    losses: list[float] = []
    ckpt_path = Path("artifacts") / "stage1_last.pt"

    if resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        bb.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                print("  [warn] optimizer state incompatible, resetting optimizer")
        start_step = ckpt.get("step", 0)
        losses = ckpt.get("losses", [])
        print(f"  Resumed at step {start_step}, continuing to {steps}")

    path = Path(openmath_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"OpenMath JSONL not found: {path}")

    row_iter = iter_jsonl(path)

    # ``opt_step_idx`` counts only completed optimizer steps (the unit the
    # paper means by "5 000 steps" with batch=2). ``micro_idx`` counts each
    # individual forward/backward, so the loop runs
    # ``steps * grad_accum_steps`` micro-iterations to perform ``steps``
    # actual SGD updates.
    opt.zero_grad(set_to_none=True)
    opt_step_idx = start_step
    micro_idx = 0
    micro_loss_accum = 0.0
    target_micro = (steps - start_step) * grad_accum_steps

    while opt_step_idx < steps and micro_idx < target_micro:
        try:
            row = next(row_iter)
        except StopIteration:
            row_iter = iter_jsonl(path)
            row = next(row_iter)

        text = prompt_text_from_openmath_row(row)
        d = bb.hidden_size
        z = torch.randn(K, d, device=dev, dtype=torch.float32) * 0.02
        w_g = bb.routing_matrix().to(device=dev, dtype=torch.float32)

        extra: Dict[str, Any] = {
            "top_k": int(cfg.get("top_k_modules", 3)),
            "deq_map_mode": bb.deq_map_mode,
        }

        lambda_lm = float(cfg.get("stage1_lambda_lm", 0.1))
        # Always run the forward inside bf16 autocast on CUDA so the
        # model's bf16 weights match auto-promoted activations from
        # fp32 inputs (z is fp32 to keep GradScaler-free training
        # numerically stable). When scaler is enabled (fp16 path), the
        # backward path scales the loss; with scaler=None (bf16 path)
        # the backward runs at the original loss magnitude.
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if dev.type == "cuda"
            else torch.amp.autocast("cpu", enabled=False)
        )
        with autocast_ctx:
            loss = fixed_point_surrogate_loss(
                bb, text, z, nu, w_g=w_g, extra=extra,
                lambda_lm=lambda_lm, tokenizer=tok,
            )
        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        micro_loss_accum += float(loss.detach().cpu().item())
        micro_idx += 1

        if micro_idx % grad_accum_steps != 0:
            # Still accumulating gradients for this batch.
            continue

        if scaler is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                params, float(cfg.get("max_grad_norm", 1.0))
            )
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                params, float(cfg.get("max_grad_norm", 1.0))
            )
            opt.step()
        opt.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        opt_step_idx += 1

        lv = micro_loss_accum / grad_accum_steps
        micro_loss_accum = 0.0
        losses.append(lv)
        step = opt_step_idx - 1  # back-compat alias for downstream block
        if log_every and (step + 1) % log_every == 0:
            tail = sum(losses[-log_every:]) / min(len(losses), log_every)
            cur_lr = opt.param_groups[0]["lr"]
            print(
                f"stage1 step={step + 1}/{steps} loss={lv:.6f} "
                f"avg_last={tail:.6f} lr={cur_lr:.2e}"
            )

        if save_every and (step + 1) % save_every == 0:
            _save_checkpoint(
                bb, opt, step=step + 1, total_steps=steps,
                config_name=config_name, openmath_jsonl=str(path),
                lora=lora, losses=losses, path=ckpt_path,
            )
            step_ckpt = Path("artifacts") / f"stage1_step{step + 1}.pt"
            _save_checkpoint(
                bb, opt, step=step + 1, total_steps=steps,
                config_name=config_name, openmath_jsonl=str(path),
                lora=lora, losses=losses, path=step_ckpt,
            )

    _save_checkpoint(
        bb, opt, step=steps, total_steps=steps,
        config_name=config_name, openmath_jsonl=str(path),
        lora=lora, losses=losses, path=ckpt_path,
    )
    print(f"Stage 1 complete: {steps} steps, final loss={losses[-1]:.6f}")
    return {
        "checkpoint": str(ckpt_path),
        "final_loss": losses[-1] if losses else 0.0,
        "steps": steps,
    }
