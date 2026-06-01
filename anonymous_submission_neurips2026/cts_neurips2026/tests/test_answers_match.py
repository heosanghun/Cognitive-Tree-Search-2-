"""Unit tests for cts.eval.math500.{normalize_answer, answers_match}.

These two functions decide every Table 2 accuracy cell, so they are
THE most reviewer-critical pieces of evaluation code in the repo.
A regression here silently shifts every reported number, with no
crash. Pin the contract behaviour for all the canonical answer
shapes that occur in MATH-500 / GSM8K / AIME / ARC-AGI-Text gold
answers.
"""

from __future__ import annotations

import pytest

from cts.eval.math500 import answers_match, normalize_answer


# ---------- normalize_answer ----------------------------------------------

def test_normalize_strips_whitespace_and_lowercases():
    assert normalize_answer(" Hello World ") == "helloworld"


def test_normalize_strips_boxed():
    assert normalize_answer("\\boxed{42}") == "42"


def test_normalize_strips_text_macro():
    assert normalize_answer("\\text{cm}") == "cm"


def test_normalize_strips_left_right_delimiters():
    assert normalize_answer("\\left( 3 \\right)") == "(3)"


def test_normalize_drops_thousands_separator_commas():
    assert normalize_answer("1,000") == "1000"


def test_normalize_drops_dollar_and_percent_signs():
    assert normalize_answer("$3.14") == "3.14"
    assert normalize_answer("50%") == "50"


def test_normalize_strips_degree_marker():
    assert normalize_answer("90^\\circ") == "90"
    assert normalize_answer("90^{\\circ}") == "90"


def test_normalize_strips_thinspace():
    assert normalize_answer("3\\,000") == "3000"


# ---------- answers_match ---------------------------------------------------

def test_match_exact_after_normalization():
    assert answers_match("\\boxed{42}", "42") is True


def test_match_with_thousands_separator():
    assert answers_match("1,000", "1000") is True


def test_match_with_currency_and_percent_signs():
    assert answers_match("$3.14", "3.14") is True
    assert answers_match("50%", "50") is True


def test_match_with_degree_marker_difference():
    assert answers_match("90^\\circ", "90") is True


def test_match_with_boxed_and_braces():
    assert answers_match("\\boxed{3}", "3") is True


def test_no_match_when_substantively_different():
    assert answers_match("3", "4") is False
    assert answers_match("yes", "no") is False


def test_match_numeric_fallback_for_decimal_vs_int():
    # 3 vs 3.0 should match via the numeric fallback path
    assert answers_match("3", "3.0") is True
    assert answers_match("3.000", "3") is True


def test_match_numeric_fallback_extracts_leading_number():
    # "answer is 42" -> normalized -> still has letters; numeric fallback
    # should pick out 42 and match it against "42".
    # NOTE: normalize_answer doesn't strip "answer is" so this exercises the
    # numeric fallback after normalization.
    assert answers_match("answer42", "42") is True


def test_no_match_when_numbers_differ_slightly():
    # Numeric fallback uses 1e-6 tolerance; 3 vs 3.001 should NOT match.
    assert answers_match("3.001", "3.0") is False


def test_match_negative_numbers():
    assert answers_match("-5", "-5") is True
    assert answers_match("-5.0", "-5") is True


def test_match_handles_empty_strings_safely():
    # "" vs "" is technically equal after normalization; the contract is
    # that this is True (degenerate but not a crash).
    assert answers_match("", "") is True


def test_match_handles_whitespace_only():
    # "   " normalizes to "" which equals normalize("") == ""
    assert answers_match("   ", "") is True


def test_no_match_when_letter_answers_differ():
    # ARC-AGI-Text path: A/B/C/D
    assert answers_match("A", "A") is True
    assert answers_match("a", "A") is True   # case-insensitive normalization
    assert answers_match("A", "B") is False


def test_match_strips_text_units():
    # "\\text{seconds}5" vs "5" — normalize strips \\text{...}
    assert answers_match("\\text{seconds}5", "5") is True


def test_match_with_left_right_delimiters_in_one_side_only():
    assert answers_match("\\left(3\\right)", "(3)") is True


# ---------- regression: real MATH-500 / AIME shapes ------------------------

def test_match_aime_three_digit_integer():
    assert answers_match("125", "125") is True
    assert answers_match("0", "0") is True
    # AIME answers are 0-999
    assert answers_match("999", "999") is True


def test_match_math500_sqrt_expression():
    # When both sides have the same LaTeX form, exact-string match wins.
    assert answers_match("3\\sqrt{13}", "3\\sqrt{13}") is True


def test_no_match_math500_sqrt_with_different_coefficient():
    assert answers_match("2\\sqrt{13}", "3\\sqrt{13}") is False
