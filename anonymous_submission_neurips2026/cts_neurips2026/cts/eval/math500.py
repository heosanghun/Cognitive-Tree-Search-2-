"""
MATH-500 style evaluation hook.

Wire a dataset JSONL (e.g. HuggingFace `HuggingFaceH4/MATH-500` export) with fields:
  `problem` or `question`, `answer` (final boxed or numeric), optional `id`.

Grader: for a minimal path, compare `normalize_answer(pred) == normalize_answer(gold)`.
Full sympy integration can be added later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    # Apply LaTeX macro substitutions FIRST so the closing "}" in
    # patterns like \text{cm} and ^{\circ} is still present when the
    # regex tries to match. (Earlier code did rstrip("}") here, which
    # silently broke both patterns; that bug was caught by
    # tests/test_answers_match.py and is fixed now.)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    s = s.replace("\\,", "")
    s = s.replace("^\\circ", "")
    s = s.replace("^{\\circ}", "")
    # Now handle \boxed{...} -- prefer balanced extraction; fall back to
    # the legacy "strip prefix and trailing brace" only if extraction
    # finds nothing.
    if "\\boxed{" in s:
        boxed = _extract_boxed(s)
        if boxed:
            s = boxed
        else:
            s = s.replace("\\boxed{", "").rstrip("}")
    s = re.sub(r"[$%]", "", s)
    s = s.replace(",", "")
    return s


def answers_match(pred: str, gold: str) -> bool:
    """Compare answers with both string and numeric matching."""
    pn = normalize_answer(pred)
    gn = normalize_answer(gold)
    if pn == gn:
        return True
    try:
        pv = float(re.search(r"-?[\d.]+", pn).group())
        gv = float(re.search(r"-?[\d.]+", gn).group())
        if abs(pv - gv) < 1e-6:
            return True
    except (AttributeError, ValueError, TypeError):
        pass
    return False


def _extract_boxed(text: str) -> str:
    """Extract content of the last \\boxed{...} with balanced braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else text[start:]


def extract_gold(gold: str) -> str:
    """Extract final answer from gold, handling GSM8K '#### <num>' format."""
    gsm_match = re.search(r"####\s*(.+?)$", gold, re.MULTILINE)
    if gsm_match:
        return gsm_match.group(1).strip()
    return gold


def extract_answer(text: str) -> str:
    """Extract the final answer from model-generated text.

    Handles: \\boxed{...}, '#### <num>' (GSM8K), 'the answer is ...', last number.
    """
    text = text.strip()

    gsm_match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if gsm_match:
        return gsm_match.group(1).strip()

    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()

    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)",
        r"(?:answer|result)\s*[=:]\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if numbers:
        return numbers[-1]

    lines = text.strip().split("\n")
    if lines:
        last = lines[-1].strip()
        if len(last) < 100:
            return last

    return text


def load_math_samples(path: Path | str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_pass_at_1(
    samples: List[Dict[str, Any]],
    predict_fn: Callable[[str], str],
    *,
    gold_key: str = "answer",
    question_key: str = "problem",
    include_items: bool = False,
    pred_max_chars: int = 4096,
) -> Dict[str, Any]:
    """`predict_fn(question)` returns model string; compared to `gold_key` after normalize."""
    ok = 0
    n = 0
    items: List[Dict[str, Any]] = []
    for ex in samples:
        q = ex.get(question_key) or ex.get("question") or ""
        gold = ex.get(gold_key, "")
        if not q:
            continue
        pred = predict_fn(str(q))
        n += 1
        match = normalize_answer(str(pred)) == normalize_answer(str(gold))
        if match:
            ok += 1
        if include_items:
            rid = ex.get("unique_id", ex.get("id", ""))
            items.append(
                {
                    "id": str(rid) if rid is not None else "",
                    "match": match,
                    "gold": str(gold)[:512],
                    "pred": str(pred)[:pred_max_chars],
                }
            )
    out: Dict[str, Any] = {
        "pass_at_1": (ok / n) if n else 0.0,
        "n": n,
        "correct": ok,
    }
    if include_items:
        out["items"] = items
    return out


def evaluate_stub() -> dict:
    return {
        "pass_at_1": None,
        "note": "Set MATH500_JSONL and run evaluate_pass_at_1(load_math_samples(path), predict_fn)",
    }
