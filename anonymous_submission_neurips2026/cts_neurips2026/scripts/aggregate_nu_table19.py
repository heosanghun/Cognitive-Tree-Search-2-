#!/usr/bin/env python3
"""Aggregate per-step ν traces into paper Table 19.

Reads the per-problem JSONL traces emitted by
``scripts/run_cts_eval_full.py --nu-trace-dir <dir>`` (one file per
``(method, benchmark, seed)``), folds them into the paper Table 19 wide
summary, and writes a Markdown table that can be pasted into
``REVIEWER_FAQ.md`` / appendix.

Usage::

    python scripts/aggregate_nu_table19.py \\
        --runs results/p0_quick_aime results/table2_re3 \\
        --out  results/table19/nu_stats.md

Each ``--runs`` directory is searched recursively for ``*.jsonl`` files
whose lines carry a ``nu_trace`` field. Files without ``nu_trace`` are
silently skipped (so it is safe to mix ``table2_results.json`` runs that
predate the persistence hook with newer ν-traced runs).

Idempotent: re-runs overwrite the output file with the latest summary.
If no ``nu_trace`` is found, the script exits 0 with a clear message and
the renderer writes a "no data" banner — never a crash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cts.eval.nu_stats import (
    DEFAULT_DOMAIN_MAP,
    aggregate_nu_traces,
    render_table19_markdown,
    summarize_table19,
)


def _discover_jsonl(run_dirs: List[Path]) -> List[Path]:
    """Recursively collect every ``*.jsonl`` under each run directory.

    Reviewer-friendly: also accepts a single ``.jsonl`` file in place of a
    directory (so smoke runs that drop one file in CWD still work).
    """
    out: List[Path] = []
    for d in run_dirs:
        d = Path(d)
        if d.is_file() and d.suffix == ".jsonl":
            out.append(d)
            continue
        if not d.exists():
            print(f"[warn] run dir does not exist: {d}", file=sys.stderr)
            continue
        out.extend(sorted(d.rglob("*.jsonl")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate ν JSONL traces into paper Table 19 Markdown.",
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="One or more run directories (or .jsonl files) to scan recursively.",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output Markdown path. Parent dirs are created automatically.",
    )
    parser.add_argument(
        "--bonferroni-n", type=int, default=2,
        help="Bonferroni family size for the two paper-highlighted directional "
             "claims. Default: 2 (matches paper Table 19).",
    )
    args = parser.parse_args()

    jsonl_paths = _discover_jsonl([Path(d) for d in args.runs])
    print(f"[aggregate_nu_table19] scanning {len(jsonl_paths)} JSONL files")

    df = aggregate_nu_traces(jsonl_paths, DEFAULT_DOMAIN_MAP)
    if df.empty:
        print(
            "[aggregate_nu_table19] no `nu_trace` payload found in any JSONL "
            "under the provided run directories.\n"
            "  Hint: re-run an evaluation with `--nu-trace-dir <dir>` so the "
            "CTS dispatcher persists per-problem ν traces, then re-invoke "
            "this script.",
            file=sys.stderr,
        )
        # Always render a banner Markdown file so downstream pipelines that
        # expect the artifact to exist can still proceed without error.
        render_table19_markdown(df, args.out)
        print(f"[aggregate_nu_table19] wrote no-data banner -> {args.out}")
        return 0

    summary = summarize_table19(df, bonferroni_n=args.bonferroni_n)
    render_table19_markdown(summary, args.out)
    print(
        f"[aggregate_nu_table19] aggregated {len(df)} per-step rows across "
        f"{df['method'].nunique()} method(s) and {df['domain'].nunique()} "
        f"domain(s); wrote -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
