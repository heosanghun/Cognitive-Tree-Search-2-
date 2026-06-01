"""Extract prompt text from OpenMathInstruct JSONL rows (streaming JSON).

Schema-tolerant by design: the v2 corpus (paper §6.1, default download
target) uses ``problem``; the v1 corpus uses ``question``. Both are
handled, so a one-line switch in ``download_experiment_data.py`` between
``-1`` and ``-2`` does not require any change to the training pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def prompt_text_from_openmath_row(row: Dict[str, Any]) -> str:
    """
    Schema-tolerant prompt extraction:
      - OpenMathInstruct-1: ``question``
      - OpenMathInstruct-2 (paper canonical): ``problem``
    Falls back through ``instruction`` / ``prompt`` / ``input`` / ``query``
    and finally a serialized ``messages`` list before giving up.
    """
    if not isinstance(row, dict):
        return str(row)[:8192]
    # Try the v2 (paper canonical) key first, then v1, then generic fallbacks.
    for key in ("problem", "question", "instruction", "prompt", "input", "query"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if isinstance(m, dict) and m.get("content"):
                parts.append(str(m["content"]))
        if parts:
            return "\n".join(parts)[:8192]
    try:
        return json.dumps(row, ensure_ascii=False)[:8192]
    except Exception:
        return str(row)[:8192]
