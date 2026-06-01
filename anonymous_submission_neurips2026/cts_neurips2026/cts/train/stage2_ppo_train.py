"""
Stage 2: MATH JSONL prompts + GemmaCTSBackbone + MetaPolicy + PPO (clipped surrogate + value + entropy).

Rollout: encode prompt -> meta policy samples branch index -> `transition()` -> scalar reward
(`default_transition_reward` or critic-based via `use_critic_reward`).

Defaults to `CTS_DEQ_MAP_MODE=blend` for tractable local training; use `parallel_map=True` for paper-style inner map.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.deq.transition import transition
from cts.mcts.critic_reward import make_critic_reward_fn
from cts.mcts.episode import default_transition_reward
from cts.rewards.shaping import paper_reward
from cts.model.gemma_loader import load_gemma4_e4b
from cts.policy.meta_policy import MetaPolicy
from cts.train.jsonl_iter import iter_jsonl
from cts.train.ppo_core import compute_gae, ppo_clipped_loss, value_loss
from cts.types import RuntimeBudgetState
from cts.utils.config import load_config
from cts.utils.repro_seed import apply_global_seed


class ValueHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.float()).squeeze(-1)


def _math_prompt(row: Dict[str, Any]) -> str:
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()
    return str(row)[:8192]


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_stage2_math_ppo(
    *,
    math_prompts_jsonl: Path | str,
    config_name: str = "default",
    total_steps: Optional[int] = None,
    device: Optional[str] = None,
    W: int = 3,
    K: int = 64,
    # Paper §6.2 / Table 4: collect_batch (rollout buffer) = 64, ppo_epochs = 4.
    # The previous defaults (4, 2) were a smoke-test setting that silently
    # under-trained the meta-policy by 16x in samples-per-update terms and
    # broke paper-parity (NeurIPS Apr 2026 audit P0-4).
    collect_batch: Optional[int] = None,
    ppo_epochs: Optional[int] = None,
    broyden_max_iter: int = 12,
    parallel_map: bool = False,
    stage1_checkpoint: Optional[Path | str] = None,
    use_critic_reward: bool = False,
    log_every: int = 5,
    # Save an intermediate checkpoint every `save_every` steps so a
    # mid-run crash (OOM, power loss, manual kill) does not lose the
    # full ~12 GPU-h investment. Reviewers reproducing on weaker
    # hardware can resume from the latest intermediate. Default is
    # 1000 -> 10 saves over a 10 000-step retrain. Set to 0 to disable.
    save_every: int = 1000,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = load_config(config_name)
    apply_global_seed()
    deq_from_cfg = cfg.get("cts_deq_map_mode")
    if deq_from_cfg and not os.environ.get("CTS_DEQ_MAP_MODE"):
        os.environ["CTS_DEQ_MAP_MODE"] = str(deq_from_cfg)
    steps = int(total_steps if total_steps is not None else cfg.get("stage2_total_ppo_steps", 10000))
    # CLI/explicit args override config; otherwise pull paper-parity defaults.
    if collect_batch is None:
        collect_batch = int(cfg.get("ppo_collect_batch", 64))
    if ppo_epochs is None:
        ppo_epochs = int(cfg.get("ppo_epochs", 4))
    if os.environ.get("CTS_STAGE2_SMOKE"):
        steps = min(steps, 32)

    dev_s = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_s)
    map_gpu = dev_s if dev_s.startswith("cuda") else None

    if parallel_map:
        os.environ["CTS_DEQ_MAP_MODE"] = "parallel"
    else:
        os.environ.setdefault("CTS_DEQ_MAP_MODE", "blend")

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=map_gpu if map_gpu else "auto",
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)
    # Paper §6.1: Stage 1 trains a LoRA adapter (r=8, α=16, dropout=0.05)
    # on q/v/o_proj. The resulting state-dict has ``base.weight`` /
    # ``lora_A.weight`` / ``lora_B.weight`` keys for those modules. To
    # round-trip Stage 1's adaptation into Stage 2 (which operates on a
    # frozen backbone), the same LoRA structure must be present *before*
    # ``load_state_dict``; otherwise every ``base.weight`` /
    # ``lora_A.weight`` / ``lora_B.weight`` key is silently dropped as
    # an unexpected key (``strict=False``) and Stage 2 effectively
    # starts from the un-adapted Gemma backbone, which is not what the
    # paper specifies. The corresponding ``q_proj.weight`` keys would
    # also be missing on the receiver side, so the original Gemma
    # weights stay in place. We apply LoRA defensively whenever a
    # Stage 1 checkpoint is provided; the adapter is harmless when the
    # checkpoint happens to lack LoRA keys (it stays at its zero-B
    # init, i.e. identity).
    if stage1_checkpoint:
        ck = _load_torch(Path(stage1_checkpoint))
        sd = ck.get("backbone_state_dict", ck)
        if any(k.endswith("lora_A.weight") or k.endswith("lora_B.weight") for k in sd):
            from cts.train.lora_compat import apply_paper_lora

            apply_paper_lora(
                bb,
                rank=int(cfg.get("lora_rank", 8)),
                target_modules=tuple(cfg.get("lora_target", ["q_proj", "v_proj", "o_proj"])),
                dropout=0.05,
                require_match=True,
                verbose=True,
            )
        missing, unexpected = bb.load_state_dict(sd, strict=False)
        # Surface obvious mismatches so the operator can notice silently
        # dropped Stage 1 work; trim to a small head to keep logs sane.
        if missing:
            print(f"[stage2] load_state_dict missing keys (showing 5/{len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[stage2] load_state_dict unexpected keys (showing 5/{len(unexpected)}): {unexpected[:5]}")

    for p in bb.parameters():
        p.requires_grad = False
    bb.eval()

    H = bb.hidden_size
    meta = MetaPolicy(text_dim=H, hidden=256, W=W).to(dev)
    value_head = ValueHead(H).to(dev)
    critic_z = NeuroCritic(H).to(dev)

    # Paper Table 4 Stage 2: actor (MetaPolicy) lr=3e-5, critic (value head /
    # NeuroCritic) lr=1e-4. Previously a single AdamW group used the actor lr
    # for the critic too, which under-trained the value baseline and inflated
    # PPO advantage variance (NeurIPS Apr 2026 audit P0-4).
    actor_lr = float(cfg.get("ppo_lr", cfg.get("lr", 3e-5)))
    critic_lr = float(cfg.get("critic_lr", 1e-4))
    actor_params = list(meta.parameters())
    critic_params = list(value_head.parameters())
    if use_critic_reward:
        critic_params += list(critic_z.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": actor_params, "lr": actor_lr},
            {"params": critic_params, "lr": critic_lr},
        ]
    )
    train_params = actor_params + critic_params

    clip = float(cfg.get("ppo_clip_epsilon", 0.2))
    vf_coef = float(cfg.get("value_loss_coef", 0.5))
    ent_coef = float(cfg.get("entropy_coef", 0.01))
    tau_budget = float(cfg.get("tau_flops_budget", 1e14))
    lambda_halt = float(cfg.get("act_halting_penalty", 0.05))
    gae_gamma = float(cfg.get("discount_gamma", 0.99))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    K_cfg = cfg.get("latent_tokens_K")
    if K_cfg is not None and K == 64:
        K = int(K_cfg)

    path = Path(math_prompts_jsonl)
    if not path.is_file():
        raise FileNotFoundError(path)

    lines: List[Dict[str, Any]] = list(iter_jsonl(path))
    if not lines:
        raise RuntimeError(f"Empty JSONL: {path}")

    reward_fn = None
    if use_critic_reward:
        reward_fn = make_critic_reward_fn(critic_z, z_dim=H, device=dev)

    history_loss: List[float] = []
    idx = 0

    for global_step in range(steps):
        batch_obs: List[torch.Tensor] = []
        batch_actions: List[int] = []
        batch_old_logp: List[float] = []
        batch_rewards: List[float] = []
        batch_values: List[float] = []
        batch_z: List[torch.Tensor] = []

        for _ in range(collect_batch):
            row = lines[idx % len(lines)]
            idx += 1

            prompt = _math_prompt(row)
            with torch.no_grad():
                ctx = bb.encode_context(prompt)
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            obs = ctx.to(dev).float()

            with torch.no_grad():
                nu, logits = meta.logits_and_nu(obs)
                dist_old = Categorical(logits=logits)
                action = int(dist_old.sample().item())
                old_logp = float(dist_old.log_prob(torch.tensor(action, device=dev)).item())
                v_old = float(value_head(obs).item())

            budget = RuntimeBudgetState()
            tr = transition(
                prompt,
                action,
                nu,
                budget,
                bb,
                K=K,
                d=H,
                broyden_max_iter=broyden_max_iter,
                tau_flops_budget=tau_budget,
                max_decode_tokens=1,
            )
            if reward_fn is not None:
                r = reward_fn(tr)
            else:
                converged = tr.solver_stats.get("converged", False)
                depth_T = tr.budget.terminal_depth if tr.budget else 1
                r = paper_reward(correct=converged, terminal_depth=depth_T, lambda_halt=lambda_halt)

            zs = tr.z_star_child
            if zs is not None:
                vflat = zs.mean(dim=0).detach().float().reshape(-1)
                if vflat.numel() >= H:
                    batch_z.append(vflat[:H].to(dev))
                else:
                    pad = torch.zeros(H, device=dev)
                    pad[: vflat.numel()] = vflat.to(dev)
                    batch_z.append(pad)
            else:
                batch_z.append(torch.zeros(H, device=dev))

            batch_obs.append(obs.squeeze(0))
            batch_actions.append(action)
            batch_old_logp.append(old_logp)
            batch_rewards.append(r)
            batch_values.append(v_old)

        obs_stacked = torch.stack(batch_obs, dim=0)
        actions_t = torch.tensor(batch_actions, device=dev, dtype=torch.long)
        old_logp_t = torch.tensor(batch_old_logp, device=dev, dtype=torch.float32)
        rewards_t = torch.tensor(batch_rewards, device=dev, dtype=torch.float32)
        values_t = torch.tensor(batch_values, device=dev, dtype=torch.float32)

        dones_list = [True] * len(batch_rewards)
        adv_list, ret_list = compute_gae(
            batch_rewards, batch_values, dones_list, gamma=gae_gamma, lam=gae_lam,
        )
        advantages = torch.tensor(adv_list, device=dev, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = torch.tensor(ret_list, device=dev, dtype=torch.float32)

        z_batch = torch.stack(batch_z, dim=0)
        for _ in range(ppo_epochs):
            h = meta.act(meta.enc(obs_stacked))
            logits_new = meta.head_prior(h)
            dist_new = Categorical(logits=logits_new)
            new_logp = dist_new.log_prob(actions_t)
            ent = dist_new.entropy().mean()

            adv_t = advantages.detach()
            p_loss = ppo_clipped_loss(new_logp, old_logp_t, adv_t, clip=clip)
            v_pred = value_head(obs_stacked)
            v_l = value_loss(v_pred, returns)

            if use_critic_reward:
                v_z = critic_z(z_batch).squeeze(-1)
                c_l = F.mse_loss(torch.sigmoid(v_z), rewards_t.detach())
            else:
                c_l = torch.zeros((), device=dev)

            loss = p_loss + vf_coef * v_l - ent_coef * ent + 0.25 * c_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, float(cfg.get("max_grad_norm", 1.0)))
            opt.step()

        history_loss.append(float(loss.detach().cpu().item()))
        if log_every and (global_step + 1) % log_every == 0:
            print(
                f"stage2 step={global_step + 1}/{steps} loss={history_loss[-1]:.4f} "
                f"reward_mean={float(rewards_t.mean().item()):.4f}",
                flush=True,
            )

        # ---- Periodic intermediate checkpoint -----------------------
        # Saved to a *separate* path so a partial save never clobbers
        # the canonical final ckpt. Reviewers / future operators can
        # resume from this if the main run is interrupted.
        if save_every and (global_step + 1) % save_every == 0 and (global_step + 1) < steps:
            inter = Path("artifacts") / "stage2_meta_value.intermediate.pt"
            inter.parent.mkdir(parents=True, exist_ok=True)
            _save_stage2_checkpoint(
                inter,
                meta=meta,
                critic_z=critic_z,
                value_head=value_head,
                config_name=config_name,
                W=W,
                H=H,
                step=global_step + 1,
                total_steps=steps,
                collect_batch=collect_batch,
                ppo_epochs=ppo_epochs,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                lambda_halt=lambda_halt,
            )
            print(f"[stage2] intermediate ckpt @ step {global_step + 1} -> {inter}", flush=True)

    out = Path("artifacts") / "stage2_meta_value.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_stage2_checkpoint(
        out,
        meta=meta,
        critic_z=critic_z,
        value_head=value_head,
        config_name=config_name,
        W=W,
        H=H,
        step=steps,
        total_steps=steps,
        collect_batch=collect_batch,
        ppo_epochs=ppo_epochs,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        lambda_halt=lambda_halt,
    )
    print("Wrote", out)
    return {"checkpoint": str(out), "steps": steps}


def _save_stage2_checkpoint(
    path: Path,
    *,
    meta,
    critic_z,
    value_head,
    config_name: str,
    W: int,
    H: int,
    step: int,
    total_steps: int,
    collect_batch: int,
    ppo_epochs: int,
    actor_lr: float,
    critic_lr: float,
    lambda_halt: float,
) -> None:
    """Persist a Stage 2 checkpoint with both legacy and canonical
    state-dict keys *and* a structured ``training_meta`` block that
    ``scripts/run_post_stage2_pipeline.py:phase_verify_stage2`` can
    use to confirm paper-faithful hyperparameters.

    Layout:

      meta_policy_state_dict / critic_state_dict / value_head_state_dict
          NeurIPS-snapshot canonical keys.

      meta / critic_z / value_head
          Legacy keys (kept so older eval scripts continue to load).

      training_meta = {
          "step", "total_steps", "collect_batch", "ppo_epochs",
          "actor_lr", "critic_lr", "lambda_halt",
          "paper_faithful_p0_4",
      }
          Reviewer-facing audit metadata. Phase 1 reads this and
          treats ``paper_faithful_p0_4=True`` as a hard PASS rather
          than the previous None-tolerated soft PASS.

      config_name / W / text_dim
          Compatibility with prior eval harness expectations.
    """
    _meta_sd = meta.state_dict()
    _critic_sd = critic_z.state_dict()
    _value_sd = value_head.state_dict()
    paper_faithful_p0_4 = bool(int(collect_batch) == 64 and int(ppo_epochs) == 4)
    torch.save(
        {
            "meta_policy_state_dict": _meta_sd,
            "critic_state_dict": _critic_sd,
            "value_head_state_dict": _value_sd,
            "meta": _meta_sd,
            "critic_z": _critic_sd,
            "value_head": _value_sd,
            "config_name": config_name,
            "W": W,
            "text_dim": H,
            "training_meta": {
                "step": int(step),
                "total_steps": int(total_steps),
                "collect_batch": int(collect_batch),
                "ppo_epochs": int(ppo_epochs),
                "actor_lr": float(actor_lr),
                "critic_lr": float(critic_lr),
                "lambda_halt": float(lambda_halt),
                "paper_faithful_p0_4": paper_faithful_p0_4,
            },
        },
        path,
    )
