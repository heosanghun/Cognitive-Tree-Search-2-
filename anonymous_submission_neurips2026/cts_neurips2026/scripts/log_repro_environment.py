#!/usr/bin/env python3
"""Write artifacts/REPRO_ENV.json (git, torch, CUDA, CTS_* env)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from cts.utils.repro_snapshot import write_repro_snapshot

    out = Path(os.environ.get("CTS_REPRO_ENV_OUT", str(ROOT / "artifacts" / "REPRO_ENV.json")))
    write_repro_snapshot(out, root=ROOT)
    print("Wrote", out)


if __name__ == "__main__":
    main()
