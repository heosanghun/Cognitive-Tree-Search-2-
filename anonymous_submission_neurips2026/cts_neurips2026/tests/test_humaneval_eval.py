"""Unit tests for cts.eval.humaneval (paper Table 2 HumanEval cell).

Covers extract_function_body, evaluate_humaneval_predictions in both
safe string-match mode and the actual execute mode (with trivial known-
good code so we don't need a real sandbox).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from cts.eval.humaneval import (
    evaluate_humaneval_predictions,
    execute_humaneval_test,
    extract_function_body,
    load_humaneval_jsonl,
)


# ---------- extract_function_body -------------------------------------------

def test_extract_function_body_keeps_indented_block():
    completion = textwrap.dedent(
        """
        def add(a, b):
            return a + b

        def main():
            print(add(1, 2))
        """
    ).strip()
    body = extract_function_body(completion, "add")
    assert "def add" in body
    assert "return a + b" in body
    # main() should NOT be in the body extraction
    assert "def main" not in body


def test_extract_function_body_returns_full_completion_when_signature_missing():
    completion = "x = 1\ny = 2"
    out = extract_function_body(completion, "nonexistent_function")
    assert out == completion


def test_extract_function_body_handles_function_at_top():
    completion = "def f(x):\n    return x * 2\n"
    body = extract_function_body(completion, "f")
    assert "def f" in body and "return x * 2" in body


# ---------- load_humaneval_jsonl --------------------------------------------

def test_load_humaneval_jsonl_round_trip(tmp_path: Path):
    items = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def f(x): ",
            "canonical_solution": "    return x",
            "test": "def check(f): assert f(1) == 1",
            "entry_point": "f",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def g(x): ",
            "canonical_solution": "    return x + 1",
            "test": "def check(g): assert g(1) == 2",
            "entry_point": "g",
        },
    ]
    p = tmp_path / "he.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    loaded = load_humaneval_jsonl(p)
    assert len(loaded) == 2
    assert loaded[0]["task_id"] == "HumanEval/0"
    assert loaded[1]["entry_point"] == "g"


def test_load_humaneval_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "he.jsonl"
    p.write_text(
        '{"task_id": "x", "prompt": "", "canonical_solution": "", "test": "", "entry_point": ""}\n\n\n',
        encoding="utf-8",
    )
    loaded = load_humaneval_jsonl(p)
    assert len(loaded) == 1


# ---------- evaluate_humaneval_predictions (string-match mode) ---------------

def test_evaluate_string_match_perfect_match():
    items = [{"task_id": "T0", "canonical_solution": "    return x", "prompt": "", "test": "", "entry_point": ""}]
    completions = ["    return x"]
    out = evaluate_humaneval_predictions(items, completions, execute=False)
    assert out["correct"] == 1
    assert out["total"] == 1
    assert out["accuracy"] == 1.0
    assert out["execute_mode"] is False


def test_evaluate_string_match_canonical_substring():
    items = [{"task_id": "T0", "canonical_solution": "return x", "prompt": "", "test": "", "entry_point": ""}]
    # The canonical "return x" appears as a substring of the completion
    completions = ["    return x  # comment"]
    out = evaluate_humaneval_predictions(items, completions, execute=False)
    assert out["correct"] == 1


def test_evaluate_string_match_failure():
    items = [{"task_id": "T0", "canonical_solution": "    return x", "prompt": "", "test": "", "entry_point": ""}]
    completions = ["    return y"]
    out = evaluate_humaneval_predictions(items, completions, execute=False)
    assert out["correct"] == 0
    assert out["accuracy"] == 0.0


def test_evaluate_string_match_handles_truncated_predictions():
    # When fewer completions than items, only the first len(completions) are scored
    items = [
        {"task_id": "T0", "canonical_solution": "a", "prompt": "", "test": "", "entry_point": ""},
        {"task_id": "T1", "canonical_solution": "b", "prompt": "", "test": "", "entry_point": ""},
    ]
    completions = ["a"]
    out = evaluate_humaneval_predictions(items, completions, execute=False)
    assert out["total"] == 1
    assert out["correct"] == 1


def test_evaluate_string_match_empty_canonical_does_not_count_as_match():
    items = [{"task_id": "T0", "canonical_solution": "", "prompt": "", "test": "", "entry_point": ""}]
    completions = [""]
    out = evaluate_humaneval_predictions(items, completions, execute=False)
    assert out["correct"] == 0


# ---------- execute_humaneval_test (live exec with trivial code) ------------

def test_execute_humaneval_passes_for_trivial_correct_solution():
    prompt = "def add(a, b):\n"
    completion = "    return a + b\n"
    test = "def check(add):\n    assert add(2, 3) == 5\n"
    out = execute_humaneval_test(prompt, completion, test, "add")
    assert out["passed"] is True
    assert out["error"] is None


def test_execute_humaneval_reports_assertion_failure():
    prompt = "def add(a, b):\n"
    completion = "    return a - b\n"  # buggy
    test = "def check(add):\n    assert add(2, 3) == 5\n"
    out = execute_humaneval_test(prompt, completion, test, "add")
    assert out["passed"] is False
    assert out["error"] is not None


def test_execute_humaneval_reports_syntax_error():
    prompt = "def add(a, b):\n"
    completion = "    return a + b ::: invalid\n"
    test = "def check(add):\n    pass\n"
    out = execute_humaneval_test(prompt, completion, test, "add")
    assert out["passed"] is False
    assert out["error"] is not None
