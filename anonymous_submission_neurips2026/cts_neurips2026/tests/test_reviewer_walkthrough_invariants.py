"""Lock the reviewer walkthrough at 0 MISS.

The reviewer walkthrough (``scripts/reviewer_walkthrough.py``)
parses the live REPRODUCIBILITY.md §5-pent table and prints
"OK / MISS" for each row. The script is the reviewer's
interactive entry point; if any row drifts to MISS, the
reviewer's first impression is "the author's own walkthrough
is broken", which is reputationally catastrophic.

This regression test runs the walkthrough as a subprocess
(torch-free, ~1 second) and asserts:

  1. exit code 0 (zero MISS rows),
  2. the verdict line ``Walkthrough verdict: N OK, 0 MISS``
     is present,
  3. the row count N matches the §5-pent row count exactly
     (no silent drop / add),
  4. both drilldown cells (Q14 fix, meta-policy) emit their
     expected file headers.

If §5-pent changes, this test fails immediately; the
walkthrough must be updated in lockstep.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_walkthrough() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/reviewer_walkthrough.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def _count_5pent_rows() -> int:
    src = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    m = re.search(r"###\s*5-pent\..*?\n(.*?)(?=\n##\s|\Z)", src, re.DOTALL)
    assert m is not None
    body = m.group(1)
    n = 0
    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[0].startswith(":") or cells[0] in ("Paper section", ""):
            continue
        if all(set(c) <= set(":-") for c in cells):
            continue
        n += 1
    return n


def test_walkthrough_exits_zero():
    rc, out = _run_walkthrough()
    assert rc == 0, (
        f"walkthrough returned non-zero exit code {rc}; "
        f"this means at least one §5-pent row failed to resolve.\n"
        f"Last 500 chars of output:\n{out[-500:]}"
    )


def test_walkthrough_zero_miss():
    _, out = _run_walkthrough()
    m = re.search(r"Walkthrough verdict:\s*(\d+)\s*OK,\s*(\d+)\s*MISS", out)
    assert m is not None, (
        f"walkthrough output missing verdict line:\n{out[-500:]}"
    )
    miss = int(m.group(2))
    assert miss == 0, (
        f"walkthrough reports {miss} MISS rows; every paper claim "
        f"must resolve to a real file + test.\n"
        f"Last 800 chars:\n{out[-800:]}"
    )


def test_walkthrough_row_count_matches_5pent():
    """The row count printed by the walkthrough must equal the
    row count of §5-pent in REPRODUCIBILITY.md (no silent
    drop / add by the parser)."""
    _, out = _run_walkthrough()
    m = re.search(r"Walkthrough verdict:\s*(\d+)\s*OK,\s*(\d+)\s*MISS", out)
    assert m is not None
    ok = int(m.group(1))
    miss = int(m.group(2))
    seen = ok + miss
    expected = _count_5pent_rows()
    assert seen == expected, (
        f"walkthrough enumerated {seen} rows but §5-pent has {expected}; "
        f"row-count drift in scripts/reviewer_walkthrough.py "
        f"or REPRODUCIBILITY.md."
    )


def test_walkthrough_drills_q14_and_meta_policy():
    """The walkthrough must drill into both the Q14 fix and the
    meta-policy file headers (those are the two cells reviewers
    would expect to inspect first when validating the headline
    claim and the headline incident)."""
    _, out = _run_walkthrough()
    # Use ASCII-only substrings: ``§`` becomes ``\xa7`` -> ``?``
    # under the cp949 console code page on the author's host,
    # so we anchor on text that survives the encoding round-trip.
    assert "Q14 garbage-math fallback" in out, (
        "walkthrough must include the Q14 garbage-math drilldown"
    )
    assert "nu-vector adaptive control" in out, (
        "walkthrough must include the nu-vector (paper §4.5) drilldown"
    )
    assert "garbage_filter.py" in out, (
        "Q14 drilldown must show cts/eval/garbage_filter.py"
    )
    assert "meta_policy.py" in out, (
        "nu-vector drilldown must show cts/policy/meta_policy.py"
    )


def test_walkthrough_runs_under_2_seconds():
    """The walkthrough is the reviewer's interactive entry
    point; it must remain snappy. >2 s and reviewers will
    assume it's hung."""
    import time
    t0 = time.time()
    rc, _ = _run_walkthrough()
    elapsed = time.time() - t0
    assert rc == 0
    assert elapsed < 5.0, (
        f"walkthrough took {elapsed:.2f}s; should stay <5s "
        f"(target <2s, hard ceiling 5s for cold subprocess starts)"
    )
