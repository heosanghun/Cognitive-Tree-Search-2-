"""Tests for GSM8K evaluation module (paper Table 2)."""

from cts.eval.gsm8k import extract_gsm8k_answer, normalize_number, check_gsm8k_answer


def test_extract_answer_with_hashes():
    assert extract_gsm8k_answer("The answer is #### 42") == "42"


def test_extract_answer_with_commas():
    raw = extract_gsm8k_answer("#### 1,234")
    assert check_gsm8k_answer(raw, "1234")


def test_extract_answer_fallback():
    assert extract_gsm8k_answer("The answer is 42.5") == "42.5"


def test_normalize_integer():
    assert normalize_number("42") == "42"
    assert normalize_number("42.0") == "42"


def test_normalize_float():
    assert normalize_number("3.14") == "3.14"


def test_check_answer_correct():
    assert check_gsm8k_answer("42", "42")
    assert check_gsm8k_answer("1234", "1,234")


def test_check_answer_incorrect():
    assert not check_gsm8k_answer("41", "42")
