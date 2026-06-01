"""Unit tests for scripts/compare_to_paper_table2.py.

The compare-helper is documented in REVIEWER_FAQ.md as the canonical
way for a reviewer to audit local-vs-paper Table 2 deltas. Lock down:

  - the rendering logic produces a self-contained Markdown table even
    on partial local results (the common case during a re-run);
  - the formatting helpers handle missing data via &mdash;;
  - the gap is computed as (local mean - paper mean) in percent;
  - the paper headline constants are not silently mutated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "_compare_to_paper_table2",
        Path(__file__).resolve().parent.parent / "scripts" / "compare_to_paper_table2.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_compare_to_paper_table2"] = mod
    spec.loader.exec_module(mod)
    return mod


cmp_mod = _load_compare_module()


# ---------- formatting helpers -----------------------------------------------

def test_fmt_local_returns_dash_for_none():
    assert cmp_mod._fmt_local(None) == "&mdash;"


def test_fmt_local_renders_mean_std_n_in_percent():
    stat = {"mean": 0.412, "std": 0.018, "n_samples": 50}
    out = cmp_mod._fmt_local(stat)
    # mean*100 -> 41.2, std*100 -> 1.8
    assert "41.2" in out and "1.8" in out and "n=50" in out


def test_fmt_paper_returns_dash_for_unknown_method():
    assert cmp_mod._fmt_paper("nonexistent_method", "math500") == "&mdash;"


def test_fmt_paper_known_cell_includes_pm_when_std_positive():
    # Paper Table 2 cell: CTS-4nu / MATH-500 = 64.1 +- 0.8
    out = cmp_mod._fmt_paper("cts_4nu", "math500")
    assert "64.1" in out and "&pm;" in out and "0.8" in out


def test_fmt_paper_known_cell_drops_pm_when_std_zero():
    # paper greedy std is 0.0 -> should render "45.2" without the "&pm;0.0"
    out = cmp_mod._fmt_paper("greedy", "math500")
    assert out == "45.2"


def test_fmt_gap_returns_dash_when_local_missing():
    assert cmp_mod._fmt_gap(None, "cts_4nu", "math500") == "&mdash;"


def test_fmt_gap_returns_dash_when_paper_missing():
    stat = {"mean": 0.5, "std": 0.0, "n_samples": 10}
    assert cmp_mod._fmt_gap(stat, "nonexistent", "math500") == "&mdash;"


def test_fmt_gap_signed_delta():
    # Paper Table 2: CTS-4nu / MATH-500 = 64.1
    # local mean 0.641 -> 64.1%, paper 64.1 -> gap = +0.0
    stat_eq = {"mean": 0.641, "std": 0.0, "n_samples": 50}
    assert cmp_mod._fmt_gap(stat_eq, "cts_4nu", "math500") == "+0.0"

    # local 0.50 -> 50.0%, paper 64.1 -> gap = -14.1
    stat_low = {"mean": 0.50, "std": 0.0, "n_samples": 50}
    assert cmp_mod._fmt_gap(stat_low, "cts_4nu", "math500") == "-14.1"

    # local 0.70 -> 70.0%, paper 64.1 -> gap = +5.9
    stat_hi = {"mean": 0.70, "std": 0.0, "n_samples": 50}
    assert cmp_mod._fmt_gap(stat_hi, "cts_4nu", "math500") == "+5.9"


# ---------- end-to-end rendering --------------------------------------------

def test_render_markdown_works_on_empty_local():
    out = cmp_mod.render_markdown({}, Path("results/test/table2_results.json"))
    # All local cells render as &mdash; but paper headlines are still printed.
    assert "Paper vs Local Comparison" in out
    assert "Greedy" in out and "CTS-4nu (Ours)" in out
    assert "MATH-500" in out and "HumanEval" in out
    # Paper headline cells are populated:
    # Paper headline cells are populated (cts_4nu/math500 = 64.1)
    assert "64.1" in out


def test_render_markdown_with_partial_local_data():
    local = {
        "greedy": {"math500": {"mean": 0.40, "std": 0.0, "n_samples": 50}},
        "cts_4nu": {"math500": {"mean": 0.42, "std": 0.02, "n_samples": 50}},
    }
    out = cmp_mod.render_markdown(local, Path("results/test/table2_results.json"))
    # local rows present with the data we supplied
    assert "40.0" in out  # greedy/math500 local mean
    assert "42.0" in out  # cts_4nu/math500 local mean
    # missing cells (e.g. cts_4nu/aime) render as &mdash;
    assert "&mdash;" in out
    # gap line for cts_4nu/math500 should be -22.1 (= 42.0 - 64.1)
    assert "-22.1" in out


def test_render_markdown_includes_source_path():
    out = cmp_mod.render_markdown({}, Path("results/some_dir/table2_results.json"))
    # Markdown must cite the source for traceability
    assert "results/some_dir/table2_results.json" in out


# ---------- paper headline integrity -----------------------------------------

def test_paper_table2_constants_match_paper_section():
    # Lock down the headline numbers so they cannot drift silently.
    # Source: paper Table 2.
    pt = cmp_mod.PAPER_TABLE2
    assert pt["greedy"]["math500"] == 45.2
    assert pt["greedy"]["humaneval"] == 56.4
    assert pt["cts_4nu"]["math500"] == 64.1
    assert pt["cts_4nu"]["aime"] == 50.2
    assert pt["cts_4nu"]["humaneval"] == 69.6
    assert pt["native_think"]["arc_agi_text"] == 50.1
    assert pt["mcts_early_stop"]["gsm8k"] == 81.2


def test_paper_table2_has_all_five_benchmarks_per_method():
    expected = {"math500", "gsm8k", "aime", "arc_agi_text", "humaneval"}
    for method, bench_map in cmp_mod.PAPER_TABLE2.items():
        assert set(bench_map.keys()) == expected, f"method {method} missing benchmarks"


def test_paper_std_aligns_with_paper_table2_keys():
    # Every method/bench in PAPER_TABLE2 must have a matching std entry.
    for method, bench_map in cmp_mod.PAPER_TABLE2.items():
        assert method in cmp_mod.PAPER_STD, f"std missing for {method}"
        for bench in bench_map:
            assert bench in cmp_mod.PAPER_STD[method], f"std missing for {method}/{bench}"
