"""HumanEval benchmark evaluation (paper Table 2).

CTS target: 74.2 ± 0.6% under Iso-FLOP ≤ 10^14 MACs.

Paper §7.1: coding capability evaluated via local, offline HumanEval execution.
Must comply with security_eval.md sandbox policy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_function_body(completion: str, entry_point: str) -> str:
    """Extract the function body from a completion string."""
    lines = completion.split("\n")
    result_lines = []
    in_func = False
    for line in lines:
        if f"def {entry_point}" in line:
            in_func = True
            result_lines.append(line)
            continue
        if in_func:
            if line.strip() and not line[0].isspace() and not line.startswith("#"):
                break
            result_lines.append(line)
    return "\n".join(result_lines) if result_lines else completion


def load_humaneval_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load HumanEval JSONL (expects 'task_id', 'prompt', 'canonical_solution', 'test', 'entry_point')."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({
                "task_id": obj.get("task_id", ""),
                "prompt": obj.get("prompt", ""),
                "canonical_solution": obj.get("canonical_solution", ""),
                "test": obj.get("test", ""),
                "entry_point": obj.get("entry_point", ""),
            })
    return items


def execute_humaneval_test(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    *,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Execute a single HumanEval test case in a sandboxed environment.

    Returns dict with 'passed', 'error' fields.
    WARNING: Only run in sandboxed/isolated environment per security_eval.md.
    """
    code = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
    try:
        exec_globals: Dict[str, Any] = {}
        exec(code, exec_globals)  # noqa: S102
        return {"passed": True, "error": None}
    except Exception as e:
        return {"passed": False, "error": str(e)[:200]}


def evaluate_humaneval_predictions(
    items: List[Dict[str, Any]],
    completions: List[str],
    *,
    execute: bool = False,
) -> Dict[str, Any]:
    """Evaluate HumanEval completions.

    If execute=False, uses string matching (safe, no code execution).
    If execute=True, actually runs tests (requires sandbox).
    """
    correct = 0
    total = min(len(items), len(completions))
    details = []

    for i in range(total):
        item = items[i]
        completion = completions[i]

        if execute:
            result = execute_humaneval_test(
                item["prompt"],
                completion,
                item.get("test", ""),
                item.get("entry_point", ""),
            )
            passed = result["passed"]
        else:
            canonical = item.get("canonical_solution", "").strip()
            passed = bool(
                canonical
                and (canonical in completion or completion.strip() == canonical)
            )

        if passed:
            correct += 1
        details.append({
            "idx": i,
            "task_id": item.get("task_id", ""),
            "match": passed,
            "pred": completion[:100],
        })

    accuracy = correct / max(total, 1)
    return {
        "benchmark": "humaneval",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "execute_mode": execute,
        "details": details,
    }
