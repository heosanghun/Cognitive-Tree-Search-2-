#!/usr/bin/env python3
"""
Verify CTS "final goal" (code + optional experiment artifacts).

Exit 0: all enabled checks pass.

  python scripts/verify_cts_final_goal.py
  python scripts/verify_cts_final_goal.py --check-artifacts
  python scripts/verify_cts_final_goal.py --pytest-all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _need(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pytest", action="store_true", help="Skip pytest")
    ap.add_argument("--pytest-all", action="store_true", help="Include slow-marked tests")
    ap.add_argument("--check-artifacts", action="store_true", help="Require artifacts/ pipeline outputs")
    ap.add_argument("--no-check-scripts", action="store_true", help="Skip script file existence check")
    args = ap.parse_args()

    root = _root()
    failed: list[str] = []

    if not args.no_check_scripts:
        scripts = [
            root / "scripts" / "run_paper_artifacts_pipeline.py",
            root / "scripts" / "run_stage1_openmath.py",
            root / "scripts" / "run_stage2_math_ppo.py",
            root / "scripts" / "download_experiment_data.py",
            root / "cts" / "train" / "stage1_openmath_train.py",
            root / "cts" / "train" / "stage2_ppo_train.py",
        ]
        missing = [str(p.relative_to(root)) for p in scripts if not p.is_file()]
        if missing:
            failed.append(f"missing files: {missing}")
        else:
            print("OK: core scripts present")

    if args.check_artifacts:
        art = root / "artifacts"
        expected = [
            art / "table1_cts_kv.csv",
            art / "table1_kv_measured.csv",
            art / "table2_isoflop_mock.json",
            art / "RUN_MANIFEST.json",
        ]
        miss = _need(expected)
        if miss:
            failed.append(
                "artifacts incomplete — run: python scripts/run_paper_artifacts_pipeline.py --tier quick --skip-download | missing: "
                + ", ".join(str(p.relative_to(root)) for p in miss)
            )
        else:
            print("OK: artifacts/ pipeline outputs present")

    if not args.no_pytest:
        py = sys.executable
        cmd = [py, "-m", "pytest", str(root / "tests"), "-q", "--tb=line"]
        if not args.pytest_all:
            cmd.extend(["-k", "not slow"])
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            failed.append(f"pytest failed (exit {r.returncode})")

    if failed:
        print("VERIFY FAILED:", file=sys.stderr)
        for f in failed:
            print(" ", f, file=sys.stderr)
        return 1
    print("CTS final-goal verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
