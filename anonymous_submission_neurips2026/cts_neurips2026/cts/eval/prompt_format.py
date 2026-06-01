"""
Chat / instruction formatting **without loading full model weights**.

Only loads `AutoTokenizer` (vocab + template files — much smaller than `model.safetensors`).
Use for inspecting `<|think|>` / chat_template output before a heavy `--gemma` run.
"""

from __future__ import annotations

import os
from pathlib import Path

from cts.model.gemma_loader import DEFAULT_GEMMA4_E4B_ID, ensure_hub_cache_env


def resolve_model_id_for_tokenizer(model_id: str | None = None) -> str:
    ensure_hub_cache_env()
    mid = model_id or DEFAULT_GEMMA4_E4B_ID
    if mid == DEFAULT_GEMMA4_E4B_ID:
        override = os.environ.get("CTS_GEMMA_MODEL_DIR")
        if override:
            p = Path(override).expanduser().resolve()
            if p.is_dir() and (p / "config.json").is_file():
                return str(p)
    return mid


def load_tokenizer_only(model_id: str | None = None):
    from transformers import AutoTokenizer

    mid = resolve_model_id_for_tokenizer(model_id)
    return AutoTokenizer.from_pretrained(mid, trust_remote_code=True)


def format_user_prompt_chat_string(
    user_text: str,
    *,
    model_id: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """
    Return the **string** after `apply_chat_template` (no model forward).
    Falls back to `user_text` if the tokenizer has no chat template.
    """
    tok = load_tokenizer_only(model_id)
    if not getattr(tok, "chat_template", None):
        return user_text
    messages = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return user_text
