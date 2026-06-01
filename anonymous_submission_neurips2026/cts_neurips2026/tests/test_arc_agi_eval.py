"""Unit tests for cts.eval.arc_agi_text (paper Table 2 ARC-AGI-Text cell).

Covers normalize_arc_output, load_arc_text_samples (JSONL round-trip),
and evaluate_pass_at_1_arc edge cases including empty inputs, alternate
input keys, and pred truncation.
"""

from __future__ import annotations

import json
from pathlib import Path

from cts.eval.arc_agi_text import (
    evaluate_pass_at_1_arc,
    evaluate_stub,
    load_arc_text_samples,
    normalize_arc_output,
)


# ---------- normalize_arc_output --------------------------------------------

def test_normalize_strips_leading_trailing_whitespace():
    assert normalize_arc_output("  hello  ") == "hello"


def test_normalize_collapses_internal_whitespace():
    assert normalize_arc_output("a  \t b\n c") == "a b c"


def test_normalize_handles_empty_string():
    assert normalize_arc_output("") == ""


# ---------- load_arc_text_samples -------------------------------------------

def test_load_jsonl_round_trip(tmp_path: Path):
    rows = [
        {"task_id": "T0", "input": "grid 0", "output": "ans 0"},
        {"task_id": "T1", "input": "grid 1", "output": "ans 1"},
    ]
    p = tmp_path / "arc.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    loaded = load_arc_text_samples(p)
    assert len(loaded) == 2
    assert loaded[0]["task_id"] == "T0"
    assert loaded[1]["output"] == "ans 1"


def test_load_jsonl_respects_limit(tmp_path: Path):
    p = tmp_path / "arc.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"task_id": f"T{i}", "input": "x", "output": "y"}) + "\n")
    loaded = load_arc_text_samples(p, limit=3)
    assert len(loaded) == 3


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "arc.jsonl"
    p.write_text(
        '{"task_id": "T0", "input": "x", "output": "y"}\n\n   \n',
        encoding="utf-8",
    )
    loaded = load_arc_text_samples(p)
    assert len(loaded) == 1


# ---------- evaluate_pass_at_1_arc ------------------------------------------

def test_evaluate_perfect_match():
    samples = [
        {"task_id": "T0", "input": "grid", "output": "A"},
        {"task_id": "T1", "input": "grid", "output": "B"},
    ]

    def pred(inp: str) -> str:
        return "A" if "T0" in str(inp) else "B"

    # Use deterministic input -> output mapping by injecting via input string
    samples_2 = [
        {"task_id": "T0", "input": "T0_grid", "output": "A"},
        {"task_id": "T1", "input": "T1_grid", "output": "B"},
    ]
    out = evaluate_pass_at_1_arc(samples_2, pred)
    assert out["correct"] == 2
    assert out["n"] == 2
    assert out["pass_at_1"] == 1.0


def test_evaluate_partial_match():
    samples = [
        {"task_id": "T0", "input": "x", "output": "A"},
        {"task_id": "T1", "input": "x", "output": "B"},
    ]
    out = evaluate_pass_at_1_arc(samples, lambda _: "A")
    assert out["correct"] == 1
    assert out["pass_at_1"] == 0.5


def test_evaluate_skips_examples_with_empty_input():
    samples = [
        {"task_id": "T0", "input": "", "output": "A"},
        {"task_id": "T1", "input": "x", "output": "B"},
    ]
    out = evaluate_pass_at_1_arc(samples, lambda _: "B")
    # T0 is skipped (empty input); T1 contributes 1 correct
    assert out["n"] == 1
    assert out["correct"] == 1


def test_evaluate_alternate_input_key():
    samples = [
        {"task_id": "T0", "question": "fallback-text", "output": "A"},
    ]
    out = evaluate_pass_at_1_arc(samples, lambda _: "A")
    # Even though `input` key is missing, the function falls back to `question`
    assert out["n"] == 1
    assert out["correct"] == 1


def test_evaluate_returns_zero_pass_at_1_for_empty_samples():
    out = evaluate_pass_at_1_arc([], lambda _: "A")
    assert out["pass_at_1"] == 0.0
    assert out["n"] == 0
    assert out["correct"] == 0


def test_evaluate_include_items_truncates_predictions():
    samples = [{"task_id": "T0", "input": "x", "output": "A"}]
    long_pred = "A" * 10000
    out = evaluate_pass_at_1_arc(
        samples, lambda _: long_pred, include_items=True, pred_max_chars=64
    )
    assert "items" in out
    assert len(out["items"][0]["pred"]) == 64


def test_evaluate_include_items_preserves_match_flag():
    samples = [
        {"task_id": "T0", "input": "x", "output": "A"},
        {"task_id": "T1", "input": "x", "output": "B"},
    ]
    out = evaluate_pass_at_1_arc(samples, lambda _: "A", include_items=True)
    assert out["items"][0]["match"] is True
    assert out["items"][1]["match"] is False


def test_evaluate_normalizes_whitespace_for_match():
    samples = [{"task_id": "T0", "input": "x", "output": "  A  B  "}]
    out = evaluate_pass_at_1_arc(samples, lambda _: "A B")
    # The normalize function collapses internal whitespace, so "A B" matches "A  B"
    assert out["correct"] == 1


# ---------- evaluate_stub ---------------------------------------------------

def test_evaluate_stub_returns_documented_shape():
    s = evaluate_stub()
    assert s["pass_at_1"] is None
    assert "ARC_JSONL" in s["note"]
