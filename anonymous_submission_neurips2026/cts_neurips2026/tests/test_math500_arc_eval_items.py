"""Per-example items when include_items=True (no Gemma)."""

from cts.eval.arc_agi_text import evaluate_pass_at_1_arc
from cts.eval.math500 import evaluate_pass_at_1


def test_math500_items():
    samples = [
        {"problem": "1+1", "answer": "2"},
        {"problem": "x", "answer": "3"},
    ]

    def pred(q: str) -> str:
        if "1+1" in q:
            return "2"
        return "wrong"

    r = evaluate_pass_at_1(samples, pred, include_items=True)
    assert r["n"] == 2
    assert r["correct"] == 1
    assert "items" in r
    assert len(r["items"]) == 2
    assert r["items"][0]["match"] is True


def test_arc_items():
    samples = [{"input": "a", "output": "b", "task_id": "t1"}]

    def pred(_: str) -> str:
        return "b"

    r = evaluate_pass_at_1_arc(samples, pred, include_items=True)
    assert r["items"][0]["task_id"] == "t1"
    assert r["items"][0]["match"] is True
