"""Tests for the HumanEval prompt builder in `scripts/run_cts_eval_full.py`.

Background
----------
The bare ``Output only the function body`` instruction caused Gemma 4 E4B
to interpret the trailing function signature as already-completed and to
emit ``# TODO`` / ``pass`` placeholders for &gt; 90 % of HumanEval problems
(reproducing 0 % pass@1 on the local snapshot).

The fix (committed in the same change-set as this test) asks the model
to produce the *complete* function inside a ```python ... ``` code
block, which the downstream `_extract_humaneval_completion` already
parses out.

These tests pin the prompt format so a future "simplification" cannot
regress the bug.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_build_prompt():
    spec = importlib.util.spec_from_file_location(
        "_run_cts_eval_full_for_test_prompt",
        Path(__file__).resolve().parent.parent / "scripts" / "run_cts_eval_full.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_cts_eval_full_for_test_prompt"] = mod
    spec.loader.exec_module(mod)
    return mod._build_prompt


_build_prompt = _load_build_prompt()


HUMANEVAL_SAMPLE = (
    'def add(a: int, b: int) -> int:\n'
    '    """Return a + b."""\n'
)


def test_humaneval_plain_prompt_contains_full_signature():
    out = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=False)
    assert "def add" in out
    assert '"""Return a + b."""' in out
    # No "TODO" / "implement this" placeholder leaked into the instruction
    assert "TODO" not in out


def test_humaneval_plain_prompt_does_not_say_only_body():
    """The "Output only the function body" wording was the root cause of the
    placeholder-completion regression; it must not return."""
    out = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=False)
    assert "Output only the function body" not in out
    assert "no explanation" not in out  # old instruction copy


def test_humaneval_native_think_uses_chat_template_and_asks_for_code_block():
    out = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=True)
    # Must wrap in Gemma chat template
    assert "<start_of_turn>user" in out
    assert "<end_of_turn>" in out
    assert "<start_of_turn>model" in out
    # Must ask for a python code block so the downstream extractor can find it
    assert "```python" in out
    assert "code block" in out


def test_humaneval_native_think_does_not_pre_seed_open_code_fence():
    """Pre-seeding ``\\`\\`\\`python\\n`` in the model turn caused the
    extractor to fail to find the matching fence in the model output."""
    out = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=True)
    assert not out.endswith("```python\n")


def test_humaneval_plain_prompt_now_uses_chat_template():
    """ROOT_CAUSE_ANALYSIS §7: the plain greedy path was emitting bare text
    that landed out-of-distribution for the Gemma 4 E4B-it instruction-tuned
    weights, causing >90 % HumanEval problems to return `# TODO` / `pass`
    placeholders. The fix routes the plain greedy path through the same
    chat template the `native_think` path uses. Pin both behaviors here
    so a future "simplification" cannot regress the bug.
    """
    plain = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=False)
    chat = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=True)
    # Plain path now uses chat template too
    assert "<start_of_turn>user" in plain
    assert "<end_of_turn>" in plain
    assert "<start_of_turn>model" in plain
    assert "```python" in plain
    # Both variants are now identical (single source of truth)
    assert plain == chat


def test_humaneval_prompt_includes_complete_qualifier():
    """Both prompt variants should ask for the COMPLETE function (signature +
    body), not just the body."""
    plain = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=False)
    chat = _build_prompt(HUMANEVAL_SAMPLE, "humaneval", native_think=True)
    # Both variants explicitly ask for "complete function"
    assert "complete function" in plain.lower()
    assert "complete function" in chat.lower()
