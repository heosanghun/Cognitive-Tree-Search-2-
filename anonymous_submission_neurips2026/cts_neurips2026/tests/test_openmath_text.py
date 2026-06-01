from cts.train.openmath_text import prompt_text_from_openmath_row


def test_openmath_v1_question_field():
    """OpenMathInstruct-1 schema uses `question`."""
    row = {"question": " 2+2? ", "expected_answer": "4"}
    assert prompt_text_from_openmath_row(row) == "2+2?"


def test_openmath_v2_problem_field_is_canonical():
    """OpenMathInstruct-2 (paper §6.1 canonical) uses `problem`."""
    row = {"problem": "x"}
    assert prompt_text_from_openmath_row(row) == "x"


def test_openmath_v2_takes_priority_over_v1_when_both_present():
    """If both schemas are present in a row (defensive), prefer the v2
    `problem` key since v2 is the paper-canonical corpus."""
    row = {"problem": "v2-canonical", "question": "v1-legacy"}
    assert prompt_text_from_openmath_row(row) == "v2-canonical"


def test_openmath_falls_back_to_messages_list():
    """Some HF revisions store conversational turns under `messages`."""
    row = {"messages": [{"content": "hi"}, {"content": "world"}]}
    out = prompt_text_from_openmath_row(row)
    assert "hi" in out and "world" in out


def test_openmath_handles_non_dict_row():
    assert prompt_text_from_openmath_row("plain string") == "plain string"


def test_openmath_skips_empty_strings():
    row = {"problem": "", "question": "real prompt"}
    assert prompt_text_from_openmath_row(row) == "real prompt"
