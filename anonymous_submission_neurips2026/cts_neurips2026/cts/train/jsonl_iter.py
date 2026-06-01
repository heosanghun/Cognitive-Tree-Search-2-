"""Line iterator for large JSONL files (Stage1/2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def iter_jsonl(path: Path | str, *, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                break


def count_lines(path: Path | str) -> int:
    c = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c
