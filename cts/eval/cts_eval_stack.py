"""Shared CTS eval stack loader (Stage 1 LoRA + optional Stage 2 PPO heads)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.policy.meta_policy import MetaPolicy
from cts.train.lora_compat import apply_paper_lora


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_cts_backbone_with_stage1(
    model: torch.nn.Module,
    tok: Any,
    *,
    cfg: Dict[str, Any],
    stage1_ckpt: Path | str = Path("artifacts/stage1_last.pt"),
) -> GemmaCTSBackbone:
    """Wrap HF Gemma, apply LoRA shell, load Stage-1 ``backbone_state_dict``."""
    bb = GemmaCTSBackbone(model, tok)
    s1 = Path(stage1_ckpt)
    if not s1.is_file():
        print(f"  [WARN] stage1 ckpt missing at {s1}; using base Gemma backbone.", flush=True)
        bb.eval()
        return bb
    ck = _load_torch(s1)
    sd = ck.get("backbone_state_dict", ck)
    if any(k.endswith("lora_A.weight") or k.endswith("lora_B.weight") for k in sd):
        apply_paper_lora(
            bb,
            rank=int(cfg.get("lora_rank", 8)),
            target_modules=tuple(cfg.get("lora_target", ["q_proj", "v_proj", "o_proj"])),
            dropout=0.05,
            require_match=True,
            verbose=True,
        )
    bb.load_state_dict(sd, strict=False)
    bb.eval()
    return bb


def load_stage2_heads(
    *,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    device: str,
    stage2_ckpt: Path | str = Path("artifacts/stage2_meta_value.pt"),
) -> Tuple[bool, bool]:
    """Load Stage-2 meta-policy + critic weights; return (meta_ok, critic_ok)."""
    s2 = Path(stage2_ckpt)
    if not s2.is_file():
        print(
            f"  [WARN] stage2 ckpt missing at {s2} — random-init meta/critic.",
            flush=True,
        )
        return False, False
    ck = _load_torch(s2)
    map_dev = torch.device(device)
    meta_state = ck.get("meta_policy_state_dict") or ck.get("meta")
    critic_state = ck.get("critic_state_dict") or ck.get("critic_z")
    loaded_mp = meta_state is not None
    loaded_cr = critic_state is not None
    if loaded_mp:
        meta_policy.load_state_dict(meta_state, strict=False)
    if loaded_cr:
        critic.load_state_dict(critic_state, strict=False)
    meta_policy.to(map_dev).eval()
    critic.to(map_dev).eval()
    print(
        f"  [ckpt] stage2={s2.is_file()} (meta_policy={loaded_mp}, critic={loaded_cr})",
        flush=True,
    )
    return loaded_mp, loaded_cr


def eval_tau_and_timeout(cfg: Dict[str, Any]) -> Tuple[float, float]:
    tau_budget = float(cfg.get("tau_flops_budget", 1e14))
    eval_tau = min(tau_budget, float(os.environ.get("CTS_EVAL_TAU_CAP", "1e13")))
    episode_timeout_s = float(os.environ.get("CTS_EVAL_EPISODE_TIMEOUT", "180"))
    return eval_tau, episode_timeout_s
