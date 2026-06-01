"""Unit tests for cts.eval.think_prompt.

The Gemma-4-E4B-it chat template's `enable_thinking` switch is the
mechanism behind the paper's "native think" baseline (paper §7.1).
The wrapper `_apply_chat_template_safe` must:

  1. Forward `enable_thinking` ONLY when the underlying tokenizer's
     `apply_chat_template` accepts that kwarg (older HF tokenizers do
     not, and forwarding would crash with `TypeError: unexpected kwarg`).
  2. Fall back to the `conversation=` calling convention when the
     tokenizer signature requires it (some forks named the parameter
     `conversation` instead of using a positional arg).
  3. Pass `tokenize=False` and `add_generation_prompt` through.

These contracts are exercised end-to-end against mock tokenizers so we
don't need the real Gemma-4-E4B weights for the test.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from cts.eval.think_prompt import _apply_chat_template_safe


class _TokenizerWithEnableThinking:
    """Mock tokenizer whose apply_chat_template accepts enable_thinking."""

    def __init__(self) -> None:
        self.last_call: dict = {}

    def apply_chat_template(
        self,
        messages: List[dict],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        self.last_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        marker = "<|think|>" if enable_thinking else ""
        return f"{marker}USER: {messages[0]['content']}"


class _TokenizerWithoutEnableThinking:
    """Mock tokenizer whose apply_chat_template does NOT accept enable_thinking."""

    def __init__(self) -> None:
        self.last_call: dict = {}

    def apply_chat_template(
        self,
        messages: List[dict],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        self.last_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return f"USER: {messages[0]['content']}"


class _TokenizerRequiringConversationKwarg:
    """Mock tokenizer that crashes on positional `messages` arg and only
    accepts the `conversation=` kwarg (mimicking some HF forks)."""

    def __init__(self) -> None:
        self.last_call: dict = {}

    def apply_chat_template(
        self,
        conversation: Optional[List[dict]] = None,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        messages: Any = None,
    ) -> str:
        if messages is not None:
            raise TypeError("apply_chat_template() got positional 'messages'")
        if conversation is None:
            raise TypeError("apply_chat_template() missing conversation= kwarg")
        self.last_call = {
            "conversation": conversation,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        return f"USER: {conversation[0]['content']}"


def test_forwards_enable_thinking_when_supported_true():
    tok = _TokenizerWithEnableThinking()
    out = _apply_chat_template_safe(
        tok,
        [{"role": "user", "content": "Solve 2+2"}],
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert "<|think|>" in out
    assert tok.last_call["enable_thinking"] is True
    assert tok.last_call["tokenize"] is False
    assert tok.last_call["add_generation_prompt"] is True


def test_forwards_enable_thinking_when_supported_false():
    tok = _TokenizerWithEnableThinking()
    out = _apply_chat_template_safe(
        tok,
        [{"role": "user", "content": "Solve 2+2"}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "<|think|>" not in out
    assert tok.last_call["enable_thinking"] is False


def test_does_not_forward_enable_thinking_when_param_missing():
    """When the tokenizer's apply_chat_template signature does not
    accept enable_thinking, the wrapper must not try to pass it (else
    TypeError). It should still produce a valid prompt."""
    tok = _TokenizerWithoutEnableThinking()
    out = _apply_chat_template_safe(
        tok,
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True,
        enable_thinking=True,  # caller asks for thinking
    )
    # The output is the no-thinking version, but no crash
    assert out == "USER: Hi"
    assert "enable_thinking" not in tok.last_call


def test_skips_enable_thinking_when_caller_passes_none_even_if_supported():
    tok = _TokenizerWithEnableThinking()
    _ = _apply_chat_template_safe(
        tok,
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True,
        enable_thinking=None,
    )
    # When caller passes None, we shouldn't probe/forward at all.
    # Default value of enable_thinking in the mock signature is False,
    # so verify it stays False (i.e. the wrapper didn't override it).
    assert tok.last_call["enable_thinking"] is False


def test_falls_back_to_conversation_kwarg_on_typeerror():
    tok = _TokenizerRequiringConversationKwarg()
    out = _apply_chat_template_safe(
        tok,
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=False,
        enable_thinking=None,
    )
    assert out == "USER: Hi"
    assert tok.last_call["conversation"][0]["content"] == "Hi"
    assert tok.last_call["add_generation_prompt"] is False
