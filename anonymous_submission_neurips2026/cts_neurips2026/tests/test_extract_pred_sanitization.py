"""Unit tests for the chat-template sanitization step in `_extract_pred`.

The Gemma-4-E4B base tokenizer revision sometimes does not register
``<end_of_turn>`` / ``<start_of_turn>`` as special tokens, so
``tokenizer.decode(skip_special_tokens=True)`` leaves them in the raw
output. Without sanitization, predictions like
``'<end_of_turn>\\n<start_of_turn>user\\nSolve '`` were being matched
against the gold answer character-for-character and always failed.

This test pins the post-fix contract so the bug cannot regress.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_extract_pred():
    """Load `_extract_pred` from scripts/run_cts_eval_full.py without
    importing it as a top-level module (the script has heavy CUDA imports).
    """
    spec = importlib.util.spec_from_file_location(
        "_run_cts_eval_full_for_test",
        Path(__file__).resolve().parent.parent / "scripts" / "run_cts_eval_full.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_cts_eval_full_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod._extract_pred


_extract_pred = _load_extract_pred()


def test_strips_end_of_turn_token_from_math_pred():
    raw = "<end_of_turn>\n<start_of_turn>user\nSolve the problem"
    out = _extract_pred(raw, "math500")
    # The control tokens must not appear in the extracted prediction.
    assert "<end_of_turn>" not in out
    assert "<start_of_turn>" not in out


def test_strips_control_tokens_for_gsm8k_extracts_last_number():
    raw = "Sure, the answer is 42.<end_of_turn>\n<start_of_turn>user\nNext..."
    out = _extract_pred(raw, "gsm8k")
    # Must not contain control tokens AND must extract a numeric answer
    assert "<end_of_turn>" not in out
    assert "42" in out


def test_strips_control_tokens_for_arc_keeps_letter():
    raw = "<end_of_turn> The correct answer is C.<start_of_turn>user\n"
    out = _extract_pred(raw, "arc_agi_text")
    assert out == "C"


def test_truncates_at_hallucinated_next_user_turn():
    # Model emits a fresh "user\n" turn mid-completion; everything after that
    # is hallucination and must be discarded before extraction.
    raw = "The answer is 7.\nuser\nWhat is 9*9?\nThe answer is 81."
    out = _extract_pred(raw, "gsm8k")
    # Must extract 7 (the real answer), not 81 (the hallucinated one)
    assert out == "7"


def test_boxed_answer_survives_sanitization():
    raw = "The solution is \\boxed{3\\sqrt{13}}<end_of_turn>"
    out = _extract_pred(raw, "math500")
    assert out == "3\\sqrt{13}"


def test_humaneval_passes_through_raw_for_post_processing():
    # HumanEval relies on _extract_humaneval_completion downstream and must
    # NOT be sanitized at this stage (control tokens in code completions are
    # extremely unlikely and the downstream test runner needs the raw text).
    raw = "def foo(): return 1<end_of_turn>"
    out = _extract_pred(raw, "humaneval")
    assert out == raw  # passthrough


def test_empty_input_returns_empty_string():
    assert _extract_pred("", "math500") == ""
    assert _extract_pred(None, "gsm8k") == ""  # type: ignore[arg-type]


def test_handles_only_control_tokens():
    raw = "<end_of_turn><start_of_turn>"
    out = _extract_pred(raw, "math500")
    assert "<" not in out  # nothing left to extract
