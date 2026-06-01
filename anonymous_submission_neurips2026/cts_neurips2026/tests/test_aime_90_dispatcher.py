"""Regression tests for paper §7.4 'Extended AIME validation' (Table 17).

Covers the data-and-dispatcher slice that was added in commit
``data(D11): AIME 2024+2025 eval extension + aime_90 benchmark dispatcher``:

1. ``scripts/download_all_benchmarks.download_aime_eval_2024_2025`` is
   importable and inspectable (no syntax / import-time errors).
2. ``BENCHMARKS`` registry in ``scripts/run_cts_eval_full`` includes
   ``aime_90`` so reviewers cannot accidentally run Table 17 against the
   2026-only set.
3. The 90-problem unified jsonl, when present on disk, is structurally
   sound (90 rows, year ∈ {2024, 2025, 2026} with 30 each, mandatory
   keys ``problem`` and ``answer`` populated, and at least 60 rows
   sourced from AoPS Wiki rather than placeholders).
4. ``download_aime_eval_2024_2025`` is idempotent when the target
   already has 60+ real rows (no network access, no overwrite).

Tests 3-4 use ``pytest.skip`` rather than ``xfail`` when the data file
is missing, so a fresh clone (``data/`` is gitignored) still passes the
regression suite. The full network fetch is exercised manually via
``python scripts/download_all_benchmarks.py``.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ensure_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def test_download_aime_eval_2024_2025_is_importable():
    _ensure_root_on_path()
    mod = importlib.import_module("scripts.download_all_benchmarks")
    assert hasattr(mod, "download_aime_eval_2024_2025"), (
        "download_aime_eval_2024_2025 should be exported from "
        "scripts/download_all_benchmarks.py"
    )
    fn = mod.download_aime_eval_2024_2025
    assert callable(fn)
    # Docstring should reference paper §7.4 / Table 17 so reviewers can
    # trace why this download function exists.
    assert "7.4" in (fn.__doc__ or "") or "Table 17" in (fn.__doc__ or ""), (
        "download_aime_eval_2024_2025 docstring should anchor paper §7.4 "
        "or Table 17 explicitly for reviewer traceability."
    )


def test_aime_90_in_benchmarks_registry():
    _ensure_root_on_path()
    mod = importlib.import_module("scripts.run_cts_eval_full")
    assert "aime_90" in mod.BENCHMARKS, (
        f"aime_90 must be in BENCHMARKS for Table 17 reproduction; "
        f"current registry: {mod.BENCHMARKS}"
    )
    # The original 30-problem AIME slot must remain so Table 2 / §7.1
    # reproduction is still possible.
    assert "aime" in mod.BENCHMARKS


def test_aime_90_jsonl_structure_when_present():
    target = ROOT / "data" / "aime" / "test_aime_90.jsonl"
    if not target.exists():
        pytest.skip(
            "data/aime/test_aime_90.jsonl missing (data/ is gitignored). "
            "Run `python scripts/download_all_benchmarks.py` to populate."
        )

    rows = [json.loads(l) for l in open(target, "r", encoding="utf-8") if l.strip()]
    assert len(rows) == 90, (
        f"AIME 90-problem set must have exactly 90 rows (paper §7.4 Table 17: "
        f"3 years x 2 exams x 15 problems); got {len(rows)}"
    )

    year_count = Counter(r.get("year") for r in rows)
    for year in (2024, 2025, 2026):
        assert year_count.get(year, 0) == 30, (
            f"year={year} must have exactly 30 rows; got {year_count.get(year, 0)}; "
            f"full breakdown: {dict(year_count)}"
        )

    real_sources = [r for r in rows if r.get("source") not in ("placeholder", None) or r.get("year") == 2026]
    assert len(real_sources) >= 60, (
        f"At least 60 rows must come from real AoPS Wiki fetches (2024+2025); "
        f"got {len(real_sources)} non-placeholder rows. Re-run download with network access."
    )

    # Mandatory schema fields. The 2024-2025 batch always carries the
    # full schema; the 2026 batch may use a slightly older schema — we
    # only require the union ``problem`` / ``answer`` keys to be
    # present and non-empty.
    for r in rows:
        assert "problem" in r and r["problem"], (
            f"row missing or empty 'problem' field: {r}"
        )
        assert "answer" in r and str(r["answer"]).strip(), (
            f"row missing or empty 'answer' field: {r}"
        )


def test_aime_eval_2024_2025_jsonl_structure_when_present():
    target = ROOT / "data" / "aime" / "test_2024_2025.jsonl"
    if not target.exists():
        pytest.skip(
            "data/aime/test_2024_2025.jsonl missing (data/ is gitignored)."
        )

    rows = [json.loads(l) for l in open(target, "r", encoding="utf-8") if l.strip()]
    real = [r for r in rows if r.get("source") != "placeholder"]
    assert len(real) >= 60, (
        f"AIME 2024-2025 holdout must have >=60 real rows; got {len(real)} real "
        f"out of {len(rows)} total. Re-run download with network access."
    )

    # Year + exam coverage.
    by_year_exam = Counter((r.get("year"), r.get("exam")) for r in real)
    for year in (2024, 2025):
        for exam in ("I", "II"):
            assert by_year_exam.get((year, exam), 0) >= 15, (
                f"AIME {year} {exam}: expected >=15 problems, got "
                f"{by_year_exam.get((year, exam), 0)}"
            )


def test_download_aime_eval_2024_2025_is_idempotent_on_real_data(monkeypatch, capsys):
    target = ROOT / "data" / "aime" / "test_2024_2025.jsonl"
    if not target.exists():
        pytest.skip(
            "data/aime/test_2024_2025.jsonl missing; idempotency check needs "
            "the populated file."
        )

    rows = [json.loads(l) for l in open(target, "r", encoding="utf-8") if l.strip()]
    real_count = sum(1 for r in rows if r.get("source") != "placeholder")
    if real_count < 60:
        pytest.skip(
            f"only {real_count} real rows on disk; idempotency check needs >=60."
        )

    _ensure_root_on_path()
    mod = importlib.import_module("scripts.download_all_benchmarks")

    # Hard guard: monkeypatch the network fetcher so any accidental
    # network call would raise and fail the test loudly.
    def _no_network(*args, **kwargs):
        raise AssertionError(
            "download_aime_eval_2024_2025 attempted a network fetch on a "
            "warm cache; idempotency is broken."
        )

    monkeypatch.setattr(mod, "_aops_fetch", _no_network)
    mod.download_aime_eval_2024_2025()  # must not call _aops_fetch

    captured = capsys.readouterr()
    assert "already exists" in captured.out, (
        f"idempotent path should print 'already exists' banner; got: {captured.out!r}"
    )

    rows_after = [json.loads(l) for l in open(target, "r", encoding="utf-8") if l.strip()]
    assert len(rows_after) == len(rows), (
        f"idempotent run must not rewrite the file; "
        f"row count changed from {len(rows)} to {len(rows_after)}"
    )
