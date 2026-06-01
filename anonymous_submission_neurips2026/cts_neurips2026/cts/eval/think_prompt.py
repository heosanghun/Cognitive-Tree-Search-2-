"""
Gemma-4-it style `<|think|>` injection via `apply_chat_template` (see `chat_template.jinja`).

Uses **tokenizer only** for string preview; full generation still needs `--gemma` + weights.
"""

from __future__ import annotations

import inspect
from typing import List, Optional

from cts.eval.prompt_format import load_tokenizer_only, resolve_model_id_for_tokenizer


def _apply_chat_template_safe(
    tok: object,
    messages: List[dict],
    *,
    add_generation_prompt: bool,
    enable_thinking: Optional[bool],
) -> str:
    fn = getattr(tok, "apply_chat_template")
    sig = inspect.signature(fn)
    kw: dict = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if enable_thinking is not None and "enable_thinking" in sig.parameters:
        kw["enable_thinking"] = bool(enable_thinking)
    try:
        return fn(messages, **kw)
    except TypeError:
        return fn(conversation=messages, **kw)


def format_user_prompt_with_thinking(
    user_text: str,
    *,
    model_id: Optional[str] = None,
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
) -> str:
    """
    When the tokenizer chat template supports `enable_thinking`, forwards it so the
    template can emit `<|think|>` at the start (Gemma-4-E4B-it).
    """
    mid = resolve_model_id_for_tokenizer(model_id)
    tok = load_tokenizer_only(mid)
    messages = [{"role": "user", "content": user_text}]
    return _apply_chat_template_safe(
        tok,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
