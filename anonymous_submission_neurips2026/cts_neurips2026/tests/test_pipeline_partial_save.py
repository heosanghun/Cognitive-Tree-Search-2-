"""Regression tests for the post-Stage-2 pipeline partial-save patch (D-7).

Two related fixes were applied after the Apr 28 24-h-timeout incident:

1. ``scripts/run_cts_eval_full.py`` -- ``run_table2`` now writes a
   ``table2_results.partial.json`` snapshot after every (method, seed,
   benchmark) cell so a timeout / crash mid-sweep produces a salvageable
   per-cell dump rather than zero JSON.

2. ``scripts/run_post_stage2_pipeline.py`` -- ``phase_table2`` and
   ``phase_table17`` now honour ``--table2-limit`` / ``--table17-limit``
   command-line flags so reviewers (and the watcher) can run a
   compute-limited replication of the headline tables in a few hours
   rather than 24 + 8 hours.

These tests guard both fixes via static source inspection (no GPU
required, runs in the CPU-only CI lane).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_run_cts_eval_full_emits_partial_snapshot():
    """Fix #1: run_table2 must call _flush_partial after every cell."""
    src = (ROOT / "scripts" / "run_cts_eval_full.py").read_text(encoding="utf-8")
    # Snapshot path constant
    assert "table2_results.partial.json" in src, (
        "run_table2 must declare a partial-save snapshot path so "
        "timeouts produce salvageable JSON."
    )
    # Helper definition
    assert "def _flush_partial(" in src, "missing _flush_partial helper"
    # Helper invocation immediately after each acc append
    pat = re.compile(
        r"all_results\[method\]\[bench\]\.append\(acc\)\s*\n\s*_flush_partial\(\)",
    )
    assert pat.search(src), (
        "_flush_partial() must be invoked after every "
        "all_results[method][bench].append(acc) so the on-disk snapshot "
        "moves forward one cell at a time."
    )
    # Snapshot payload must include partial=True flag and raw_scores so
    # downstream pipelines can distinguish it from the canonical
    # table2_results.json.
    assert '"partial": True' in src
    assert '"raw_scores"' in src


def test_run_cts_eval_full_partial_snapshot_records_wrote_at():
    """Snapshot must record an ISO8601 UTC timestamp for audit trails."""
    src = (ROOT / "scripts" / "run_cts_eval_full.py").read_text(encoding="utf-8")
    # The actual snapshot writes the timestamp as the ``wrote_at_utc`` key
    # using the strftime formatter we wired in.  The earlier dispatcher
    # already imports ``time``, so no new import is needed.
    assert "wrote_at_utc" in src, (
        "Partial snapshot must include a wrote_at_utc field so the "
        "watcher can detect stale snapshots."
    )
    assert re.search(r"strftime\([^,]+,\s*time\.gmtime\(\)\)", src), (
        "Snapshot timestamp must be UTC (time.gmtime), not local time."
    )


def test_post_stage2_pipeline_accepts_table2_limit_kwarg():
    """Fix #2: the pipeline argparser must expose --table2-limit."""
    from scripts.run_post_stage2_pipeline import _build_argparser
    parser = _build_argparser()
    actions = {a.dest: a for a in parser._actions}
    assert "table2_limit" in actions, "missing --table2-limit option"
    assert "table17_limit" in actions, "missing --table17-limit option"
    # Defaults must be None so existing CI / production runs are unchanged.
    assert actions["table2_limit"].default is None
    assert actions["table17_limit"].default is None
    # Both must be int.
    assert actions["table2_limit"].type is int
    assert actions["table17_limit"].type is int


def test_phase_table2_appends_limit_when_set():
    """phase_table2 must append --limit when args.table2_limit is set."""
    src = (ROOT / "scripts" / "run_post_stage2_pipeline.py").read_text(encoding="utf-8")
    # The production-with-limit branch must use args.table2_limit and pass
    # it via --limit to the underlying script.
    pat = re.compile(
        r"elif\s+args\.table2_limit:\s*\n.*?cmd\.extend\(\[\"--limit\",\s*str\(args\.table2_limit\)\]\)",
        re.DOTALL,
    )
    assert pat.search(src), (
        "phase_table2 must extend the subprocess command with "
        '["--limit", str(args.table2_limit)] in its production-with-'
        "limit branch."
    )
    # Same for table17
    pat17 = re.compile(
        r"elif\s+args\.table17_limit:\s*\n.*?cmd\.extend\(\[\"--limit\",\s*str\(args\.table17_limit\)\]\)",
        re.DOTALL,
    )
    assert pat17.search(src), (
        "phase_table17 must extend the subprocess command with "
        '["--limit", str(args.table17_limit)].'
    )


def test_pipeline_help_documents_partial_save_lineage():
    """The --table*-limit help strings must reference CHANGELOG D-7 ancestry."""
    from scripts.run_post_stage2_pipeline import _build_argparser
    parser = _build_argparser()
    actions = {a.dest: a for a in parser._actions}
    h2 = actions["table2_limit"].help or ""
    assert "compute-limited" in h2.lower() or "partial-save" in h2.lower(), (
        "Help string for --table2-limit must explain the rationale "
        "(compute-limited replication / partial-save) so reviewers "
        "find the audit trail in CHANGELOG."
    )


@pytest.mark.parametrize(
    "argv,expected_limit",
    [
        (["--table2-limit", "10"], 10),
        (["--table2-limit", "5"], 5),
        ([], None),
        (["--smoke"], None),
    ],
)
def test_pipeline_table2_limit_argparse(argv, expected_limit):
    """End-to-end argparse check for the --table2-limit knob."""
    from scripts.run_post_stage2_pipeline import _build_argparser
    parser = _build_argparser()
    args = parser.parse_args(argv)
    assert args.table2_limit == expected_limit
