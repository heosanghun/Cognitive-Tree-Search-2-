"""
ARC-AGI style eval with **text-serialized** grids (paper / bench parity hook).

Expect JSONL lines with `task_id`, `input` (string grid or prompt), `output` (gold).
Grader: optional exact match after `normalize_arc_output`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def normalize_arc_output(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_arc_text_samples(path: Path | str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
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


def evaluate_pass_at_1_arc(
    samples: List[Dict[str, Any]],
    predict_fn: Callable[[str], str],
    *,
    input_key: str = "input",
    gold_key: str = "output",
    include_items: bool = False,
    pred_max_chars: int = 4096,
) -> Dict[str, Any]:
    ok = 0
    n = 0
    items: List[Dict[str, Any]] = []
    for ex in samples:
        inp = ex.get(input_key) or ex.get("question") or ""
        gold = ex.get(gold_key, "")
        if not inp:
            continue
        pred = predict_fn(str(inp))
        n += 1
        match = normalize_arc_output(str(pred)) == normalize_arc_output(str(gold))
        if match:
            ok += 1
        if include_items:
            tid = ex.get("task_id", ex.get("id", ""))
            items.append(
                {
                    "task_id": str(tid) if tid is not None else "",
                    "match": match,
                    "gold": str(gold)[:512],
                    "pred": str(pred)[:pred_max_chars],
                }
            )
    out: Dict[str, Any] = {"pass_at_1": (ok / n) if n else 0.0, "n": n, "correct": ok}
    if include_items:
        out["items"] = items
    return out


def evaluate_stub() -> dict:
    return {
        "pass_at_1": None,
        "note": "Set ARC_JSONL and run evaluate_pass_at_1_arc(load_arc_text_samples(path), predict_fn)",
    }
