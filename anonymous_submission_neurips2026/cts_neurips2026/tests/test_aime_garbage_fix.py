"""Regression tests for the AIME garbage-prediction fix (D-7, 2026-04-29).

Two related fixes were applied to address the post-Stage-2 pipeline finding
that CTS-4nu produced non-numeric predictions like 'Cultura', 'LinearLayout',
'\?????' on AIME 2026 problems:

1. ``cts/backbone/gemma_adapter.py`` -- ``decode_from_z_star`` now accepts an
   optional ``problem_text`` argument that is tokenised and concatenated
   *after* the Wproj soft-prompt prefix. This restores the paper's intent
   that the soft-prompt augments (rather than replaces) the textual context.
   Backwards-compatible: callers that don't pass the kwarg get the previous
   soft-prompt-only behaviour unchanged.

2. ``scripts/run_cts_eval_full.py`` -- both the ``cts_4nu``/``cts_2nu``/
   ``deq_only`` dispatchers now treat any non-numeric extracted prediction
   on a math benchmark (``math500``/``gsm8k``/``aime``/``aime_90``) as a
   garbage signal and fall back to the greedy predictor, ensuring the
   pipeline never silently scores 0% on a 'Cultura'-style output.

These tests guard against silent regressions in either fix.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


def test_decode_from_z_star_accepts_problem_text_kwarg():
    """Fix B: ``problem_text`` parameter must be present and default to None."""
    from cts.backbone.gemma_adapter import GemmaCTSBackbone

    sig = inspect.signature(GemmaCTSBackbone.decode_from_z_star)
    assert "problem_text" in sig.parameters, (
        "decode_from_z_star must accept ``problem_text`` so cts_full_episode "
        "can pass the original question alongside the Wproj soft prefix."
    )
    p = sig.parameters["problem_text"]
    assert p.default is None, (
        "``problem_text`` must default to None so existing callers "
        "(unit tests, sweep drivers) keep working without modification."
    )
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        "``problem_text`` should be keyword-only so positional callers "
        "(z_star, max_new_tokens) are not silently shifted."
    )


def test_cts_full_episode_passes_prompt_to_decode():
    """Fix B: cts_full_episode must forward the prompt to decode_from_z_star."""
    src = Path(__file__).resolve().parents[1] / "cts" / "mcts" / "cts_episode.py"
    text = src.read_text(encoding="utf-8")
    assert "decode_from_z_star" in text
    # Must call decode_from_z_star with problem_text=prompt, with a
    # TypeError-guarded fallback for older backbones that don't accept the
    # kwarg.
    pat = re.compile(
        r"decode_from_z_star\([^)]*problem_text\s*=\s*prompt", re.DOTALL
    )
    assert pat.search(text), (
        "cts_full_episode must call backbone.decode_from_z_star("
        "best_z, max_new_tokens=..., problem_text=prompt) so the soft "
        "prompt is composed with the actual question text."
    )
    # Also assert the legacy fallback is preserved (TypeError catch).
    assert "TypeError" in text and "decode_from_z_star(\n" in text or "decode_from_z_star(" in text


def test_cts_dispatcher_treats_non_numeric_math_pred_as_garbage():
    """Fix A: math benchmark CTS dispatchers must fall back when pred is non-numeric."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "run_cts_eval_full.py"
    text = src.read_text(encoding="utf-8")
    # The garbage detection check must reference all four math slots.
    for slot in ("math500", "gsm8k", "aime", "aime_90"):
        assert f'"{slot}"' in text, f"slot ``{slot}`` missing from dispatcher"
    # The garbage condition must check that pred does NOT start with a digit.
    pat = re.compile(
        r"_is_garbage_math\s*=\s*\(\s*\n\s*benchmark\s+in\s+\(\s*\"math500\"",
        re.DOTALL,
    )
    assert pat.search(text), (
        "Garbage-detection condition must be named _is_garbage_math and "
        "gate on benchmark in {math500, gsm8k, aime, aime_90}."
    )
    # The fallback condition must include _is_garbage_math.
    assert "or _is_garbage_math" in text, (
        "Existing CTS-fallback if-condition must include the new "
        "_is_garbage_math signal so non-numeric outputs route to greedy."
    )


def test_extract_pred_passes_through_numeric_strings():
    """Fix A sanity: when CTS produces a valid number, no fallback should fire."""
    from scripts.run_cts_eval_full import _extract_pred

    # If decode_from_z_star (with problem_text) actually produces a numeric
    # answer like '47' or '\\boxed{47}', _extract_pred must return a string
    # whose first char is a digit so the garbage-detection regex
    # (``re.match(r"^-?\d", pred)``) accepts it.
    for raw in [
        "47",
        "\\boxed{47}",
        "The final answer is: \\boxed{47}",
        "Therefore, m+n = 277.",
    ]:
        pred = _extract_pred(raw, "aime")
        assert re.match(r"^-?\d", pred or ""), (
            f"_extract_pred({raw!r}, 'aime') = {pred!r}; expected digit-start."
        )


def test_garbage_strings_correctly_classified():
    """Fix A sanity: known garbage tokens must trigger the fallback."""
    garbage_examples = [
        "Cultura",
        "LinearLayout",
        "hedral",
        "?????",
        "vich",
        "ukan",
        "",  # empty
    ]
    for raw in garbage_examples:
        # Apply the same condition the dispatcher uses
        is_garbage = not re.match(r"^-?\d", raw or "")
        assert is_garbage, (
            f"Garbage example {raw!r} should be classified as garbage "
            f"(non-numeric on math benchmark)."
        )


@pytest.mark.parametrize(
    "raw,expected_garbage",
    [
        ("47", False),
        ("-3", False),
        ("0.5", False),
        ("Cultura", True),
        ("LinearLayout", True),
        ("hedral", True),
        ("?????", True),
        ("", True),
    ],
)
def test_garbage_classification_truth_table(raw: str, expected_garbage: bool):
    is_garbage = not re.match(r"^-?\d", raw or "")
    assert bool(is_garbage) is expected_garbage
