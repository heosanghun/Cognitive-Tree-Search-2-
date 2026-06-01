"""Load Gemma 4 E4B with optional Vision/Audio offloading (paper §7.1)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

DEFAULT_GEMMA4_E4B_ID = "google/gemma-4-E4B"


def default_hub_cache_dir() -> Optional[Path]:
    """Prefer repo-local cache when HF_HUB_CACHE is unset."""
    repo_root = Path(__file__).resolve().parents[2]
    local = repo_root / ".hf_cache"
    return local if local.parent.exists() else None


def ensure_hub_cache_env() -> None:
    if os.environ.get("HF_HUB_CACHE"):
        return
    cand = default_hub_cache_dir()
    if cand is not None:
        cand.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cand))


def _offload_vision_audio(model: Any) -> float:
    """Paper §7.1: offload vision (~150M) and audio (~300M) encoders.

    Returns approximate VRAM saved in GB.
    """
    saved_params = 0
    for attr_name in ("vision_tower", "vision_model", "audio_tower", "audio_model",
                      "multi_modal_projector", "audio_encoder"):
        sub = getattr(model, attr_name, None) if hasattr(model, attr_name) else None
        if sub is None and hasattr(model, "model"):
            sub = getattr(model.model, attr_name, None)
        if sub is not None and hasattr(sub, "parameters"):
            for p in sub.parameters():
                saved_params += p.numel()
                p.data = p.data.to("cpu")
                p.requires_grad = False
    saved_gb = saved_params * 2 / (1024 ** 3)  # BF16 = 2 bytes
    return saved_gb


def load_gemma4_e4b(
    model_id: str = DEFAULT_GEMMA4_E4B_ID,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    token: Optional[str] = None,
    hub_cache: Optional[str | Path] = None,
    low_cpu_mem_usage: bool = True,
    offload_vision_audio: bool = False,
) -> Tuple[Any, Any]:
    """
    Returns (Gemma4ForConditionalGeneration, AutoTokenizer).

    Args:
        offload_vision_audio: if True, move vision/audio encoders to CPU
            saving ~0.9 GB VRAM (paper §7.1 — text-only reasoning).
    """
    ensure_hub_cache_env()
    if hub_cache is not None:
        os.environ["HF_HUB_CACHE"] = str(hub_cache)

    token = token or os.environ.get("HF_TOKEN")

    if model_id == DEFAULT_GEMMA4_E4B_ID:
        override = os.environ.get("CTS_GEMMA_MODEL_DIR")
        if override:
            p = Path(override).expanduser().resolve()
            if p.is_dir() and (p / "config.json").is_file():
                model_id = str(p)

    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        token=token,
        trust_remote_code=True,
    )

    if offload_vision_audio:
        saved = _offload_vision_audio(model)
        if saved > 0:
            print(f"[CTS] Offloaded vision/audio encoders, saved ~{saved:.2f} GB VRAM")

    return model, tokenizer
