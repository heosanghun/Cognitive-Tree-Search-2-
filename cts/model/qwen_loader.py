"""Load Qwen 2.5 7B model and tokenizer (paper Table 18)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from cts.model.gemma_loader import ensure_hub_cache_env

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-7B"


def load_qwen2_5_7b(
    model_id: str = DEFAULT_QWEN_MODEL_ID,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    token: Optional[str] = None,
    hub_cache: Optional[str | Path] = None,
    low_cpu_mem_usage: bool = True,
) -> Tuple[Any, Any]:
    """
    Returns (Qwen2ForCausalLM, AutoTokenizer).
    """
    ensure_hub_cache_env()
    if hub_cache is not None:
        os.environ["HF_HUB_CACHE"] = str(hub_cache)

    token = token or os.environ.get("HF_TOKEN")

    if model_id == DEFAULT_QWEN_MODEL_ID:
        override = os.environ.get("CTS_QWEN_MODEL_DIR")
        if override:
            p = Path(override).expanduser().resolve()
            if p.is_dir() and (p / "config.json").is_file():
                model_id = str(p)

    from transformers import AutoTokenizer, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    model = Qwen2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        token=token,
        trust_remote_code=True,
    )

    return model, tokenizer
