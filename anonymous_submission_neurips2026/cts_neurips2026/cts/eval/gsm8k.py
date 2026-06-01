"""GSM8K benchmark evaluation (paper Table 2).

CTS target: 92.1 ± 0.5% under Iso-FLOP ≤ 10^14 MACs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numeric answer from GSM8K response (#### format)."""
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1] if nums else None


def normalize_number(s: str) -> str:
    """Normalize numeric string for comparison."""
    try:
        val = float(s.replace(",", ""))
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return s.strip()


def check_gsm8k_answer(predicted: str, gold: str) -> bool:
    return normalize_number(predicted) == normalize_number(gold)


def load_gsm8k_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load GSM8K JSONL (expects 'question' and 'answer' fields)."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer", "")
            gold = extract_gsm8k_answer(a) or a
            items.append({"question": q, "answer": a, "gold": gold})
    return items


def evaluate_gsm8k_predictions(
    items: List[Dict[str, Any]],
    predictions: List[str],
) -> Dict[str, Any]:
    """Evaluate GSM8K predictions against gold answers."""
    correct = 0
    total = min(len(items), len(predictions))
    details = []
    for i in range(total):
        pred_answer = extract_gsm8k_answer(predictions[i]) or predictions[i]
        gold = items[i]["gold"]
        match = check_gsm8k_answer(pred_answer, gold)
        if match:
            correct += 1
        details.append({
            "idx": i,
            "match": match,
            "pred": pred_answer[:100],
            "gold": gold[:100],
        })
    accuracy = correct / max(total, 1)
    return {
        "benchmark": "gsm8k",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }
