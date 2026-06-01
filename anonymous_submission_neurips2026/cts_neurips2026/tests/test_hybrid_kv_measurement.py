"""Regression tests for the honest Hybrid-KV (paper §7.7) measurement
scaffold.

These tests are CPU-only and complete in <30 s on a laptop. They cover:

  - ``cts/eval/hybrid_kv_measurement.py`` (TOST equivalence, decision-
    overhead measurement, summary aggregator, Markdown rendering).
  - ``cts/eval/cuda_graph_skeleton.py`` (future-work skeleton: must
    honestly return ``False`` on ``would_capture`` and a non-empty
    planned-CLI string).
  - ``scripts/measure_hybrid_kv.py`` end-to-end via subprocess.

All tests are designed to stay green even after the cache-HIT path
lands; the assertions target the *honest* properties (caveat at top of
report, decision-overhead bounds, TOST mechanics) rather than any
particular wall-clock number.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cts.backbone.mock_tiny import MockTinyBackbone  # noqa: E402
from cts.critic.neuro_critic import NeuroCritic  # noqa: E402
from cts.eval.cuda_graph_skeleton import (  # noqa: E402
    PLANNED_CAPTURE_CLI,
    planned_capture_blockers,
    planned_capture_cli,
    would_capture,
)
from cts.eval.hybrid_kv_measurement import (  # noqa: E402
    ALL_MODES,
    KV_REUSE_CAVEAT,
    MODE_DECISION_ONLY,
    MODE_OFF,
    measure_decision_overhead,
    render_hybrid_kv_markdown,
    summarize_hybrid_kv,
    tost_equivalence,
)
from cts.policy.meta_policy import MetaPolicy  # noqa: E402


class _DecodingMockBackbone(MockTinyBackbone):
    """Mirror of the helper in ``tests/test_cts_full_episode.py``."""

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 64) -> str:
        head = z_star.detach().float().mean().item()
        return f"answer={head:+.4f}|tokens={max_new_tokens}"


def _build_components(d: int = 16, W: int = 2):
    torch.manual_seed(2026)
    bb = _DecodingMockBackbone(hidden=d, num_layers=4)
    meta = MetaPolicy(text_dim=d, hidden=32, W=W)
    critic = NeuroCritic(z_dim=d)
    return bb, meta, critic


# ---------------------------------------------------------------------------
# 1. TOST equivalence test
# ---------------------------------------------------------------------------


def test_tost_equivalence_identical_samples() -> None:
    """Two identical samples must be declared equivalent at any positive
    margin: mean_diff = 0 < delta and the degenerate-variance branch
    returns ``equivalent=True`` with both p-values 0.0.
    """
    out = tost_equivalence([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], delta=0.1)
    assert out["equivalent"] is True
    assert out["p_max"] == 0.0
    assert out["p_lower"] == 0.0
    assert out["p_upper"] == 0.0
    assert out["mean_diff"] == 0.0
    assert out["n_a"] == 4 and out["n_b"] == 4
    assert out["delta"] == 0.1


def test_tost_equivalence_far_apart() -> None:
    """Samples whose mean difference exceeds the margin must be NOT
    equivalent. Zero-variance branch keeps the math simple: mean_diff
    = -1.0, |diff| >= delta -> equivalent=False with both p-values 1.0.
    """
    out = tost_equivalence([1.0] * 8, [2.0] * 8, delta=0.1)
    assert out["equivalent"] is False
    assert out["p_max"] == 1.0
    assert out["mean_diff"] == pytest.approx(-1.0)


def test_tost_equivalence_marginally_different() -> None:
    """Same samples, two margins: a generous margin must declare
    equivalence and a tight margin must not.

    Concretely: ``a = [1.0]*8`` and ``b = [1.05]*8`` have ``mean_diff =
    -0.05``. With ``delta = 0.1`` the difference fits inside the
    margin (``equivalent=True``); with ``delta = 0.01`` it does not
    (``equivalent=False``). This pins both branches of the boundary
    rather than asserting an arbitrary single value.
    """
    a = [1.0] * 8
    b = [1.05] * 8

    wide = tost_equivalence(a, b, delta=0.1)
    assert wide["equivalent"] is True, wide
    assert wide["mean_diff"] == pytest.approx(-0.05)

    narrow = tost_equivalence(a, b, delta=0.01)
    assert narrow["equivalent"] is False, narrow
    assert narrow["mean_diff"] == pytest.approx(-0.05)

    # The verdict must flip strictly because of delta, not because the
    # samples changed (regression guard against accidentally swapping
    # arguments inside ``tost_equivalence``).
    assert wide["mean_diff"] == narrow["mean_diff"]


def test_tost_equivalence_with_variance_borderline() -> None:
    """Non-trivial variance: small samples that *do* declare
    equivalence at a comfortable margin. Sanity-checks that the Welch /
    t-CDF code path runs at all (the previous tests all hit the
    zero-variance fast path).
    """
    # Two samples drawn from the same constant signal plus tiny noise.
    a = [0.500, 0.502, 0.498, 0.501, 0.499, 0.500, 0.501, 0.499]
    b = [0.501, 0.500, 0.499, 0.502, 0.498, 0.501, 0.500, 0.499]
    out = tost_equivalence(a, b, delta=0.05)
    assert out["equivalent"] is True
    assert 0.0 <= out["p_lower"] <= 1.0
    assert 0.0 <= out["p_upper"] <= 1.0


# ---------------------------------------------------------------------------
# 2. measure_decision_overhead
# ---------------------------------------------------------------------------


def test_measure_decision_overhead_returns_long_form_df() -> None:
    """2 problems × 2 seeds × 2 modes -> 8 rows with the 7 expected columns.

    Uses the same dummy backbone pattern as
    ``tests/test_cts_full_episode.py`` so the measurement harness is
    proven to work on the canonical CPU mock without any GPU bits.
    """
    bb, meta, critic = _build_components()
    problems = [
        {"problem": "Q1: 2 + 3 = ?"},
        {"problem": "Q2: 5 - 1 = ?"},
    ]
    df = measure_decision_overhead(
        bb, problems, seeds=[0, 1], meta_policy=meta, critic=critic,
    )
    assert len(df) == 8, df
    expected_cols = {
        "seed", "problem_id", "mode", "wall_seconds",
        "decision_calls", "cached_nodes", "vram_used_gb",
    }
    assert set(df.columns) == expected_cols, df.columns

    # Every (seed, problem_id) pair must be present in BOTH modes.
    pivoted = df.pivot_table(
        index=["seed", "problem_id"], columns="mode", values="wall_seconds",
        aggfunc="size",
    )
    assert sorted(pivoted.columns.tolist()) == sorted(list(ALL_MODES))
    assert (pivoted == 1).all().all()

    # `hybrid_off` rows must have zero decision_calls / cached_nodes
    # (no manager attached at all). `hybrid_decision_only` MUST have at
    # least one decision call per row, otherwise the §7.7 wiring is
    # silently broken.
    off = df[df["mode"] == MODE_OFF]
    on = df[df["mode"] == MODE_DECISION_ONLY]
    assert (off["decision_calls"] == 0).all()
    assert (off["cached_nodes"] == 0).all()
    assert (on["decision_calls"] >= 1).all(), on


# ---------------------------------------------------------------------------
# 3. summarize_hybrid_kv
# ---------------------------------------------------------------------------


def test_summarize_hybrid_kv_returns_tost_verdict() -> None:
    """Two modes that differ by ~1% with low variance must be declared
    equivalent at the default 5% margin.
    """
    rows: List[dict] = []
    base_off = 1.000
    base_on = 1.010
    for seed in range(3):
        for pid in range(4):
            rows.append({
                "seed": seed, "problem_id": pid, "mode": MODE_OFF,
                "wall_seconds": base_off + 0.0005 * pid,
                "decision_calls": 0, "cached_nodes": 0, "vram_used_gb": 0.0,
            })
            rows.append({
                "seed": seed, "problem_id": pid, "mode": MODE_DECISION_ONLY,
                "wall_seconds": base_on + 0.0005 * pid,
                "decision_calls": 5, "cached_nodes": 0, "vram_used_gb": 0.0,
            })
    df = pd.DataFrame(rows)
    summary = summarize_hybrid_kv(df, margin_frac=0.05, alpha=0.05)

    assert summary["n_seeds"] == 3
    assert summary["n_problems"] == 4
    assert MODE_OFF in summary["by_mode"]
    assert MODE_DECISION_ONLY in summary["by_mode"]
    assert summary["by_mode"][MODE_OFF]["wall_seconds_mean"] == pytest.approx(
        base_off + 0.00075, abs=1e-6,
    )
    assert summary["tost"]["equivalent"] is True, summary
    assert summary["tost"]["delta"] > 0.0
    assert "caveat" in summary
    assert "KV-reuse hit path NOT YET measured" in summary["caveat"]


# ---------------------------------------------------------------------------
# 4. render_hybrid_kv_markdown
# ---------------------------------------------------------------------------


def test_render_hybrid_kv_markdown_includes_caveat_at_top(tmp_path: Path) -> None:
    """The rendered Markdown's first 30 lines must contain the
    canonical ``KV-reuse hit path NOT YET measured`` disclosure so
    reviewers see it BEFORE any number.
    """
    summary = {
        "by_mode": {
            MODE_OFF: {
                "n": 4, "wall_seconds_mean": 1.0, "wall_seconds_std": 0.01,
                "decision_calls_mean": 0.0, "cached_nodes_mean": 0.0, "vram_used_gb_mean": 0.0,
            },
            MODE_DECISION_ONLY: {
                "n": 4, "wall_seconds_mean": 1.005, "wall_seconds_std": 0.01,
                "decision_calls_mean": 5.0, "cached_nodes_mean": 0.0, "vram_used_gb_mean": 0.0,
            },
        },
        "tost": tost_equivalence([1.0, 1.0, 1.0, 1.0], [1.005, 1.005, 1.005, 1.005], delta=0.05),
        "n_seeds": 1,
        "n_problems": 4,
        "margin_frac": 0.05,
        "caveat": KV_REUSE_CAVEAT,
    }
    out = tmp_path / "report.md"
    render_hybrid_kv_markdown(summary, out)

    txt = out.read_text(encoding="utf-8")
    assert txt.strip(), "Markdown report is empty"

    head = "\n".join(txt.splitlines()[:30])
    assert "KV-reuse hit path NOT YET measured" in head, head
    # The headline section title must appear before the table so a
    # casual reader sees the disclosure first.
    head_idx = txt.find("KV-reuse hit path NOT YET measured")
    table_idx = txt.find("| `hybrid_off`")
    assert head_idx < table_idx, (head_idx, table_idx)


# ---------------------------------------------------------------------------
# 5. cuda_graph_skeleton — future-work scaffold
# ---------------------------------------------------------------------------


def test_cuda_graph_skeleton_would_capture_returns_false() -> None:
    """The honest answer today is False; the function must not pretend.

    A True return would imply the L-Broyden inner loop has been
    refactored into a static, capturable variant — which it has not
    (see ``cts/deq/broyden_forward.py`` lines 106-173).
    """
    assert would_capture(None) is False
    # The function must accept arbitrary state without raising — the
    # signature is intentionally forward-compatible with the future
    # implementation.
    assert would_capture({"n": 4096, "memory_limit": 16}) is False
    # Multiple blockers documented honestly.
    blockers = planned_capture_blockers()
    assert isinstance(blockers, list) and len(blockers) >= 3
    assert all(isinstance(b, str) and b for b in blockers)


def test_cuda_graph_skeleton_planned_capture_cli_is_nonempty() -> None:
    """The planned CLI string must exist, be non-empty, and reference
    ``torch.cuda.graph`` (or the planned ``--enable-cuda-graph`` flag)
    so reviewers can locate the future entry-point with grep.
    """
    s = planned_capture_cli()
    assert isinstance(s, str)
    assert len(s.strip()) > 0
    lower = s.lower()
    assert (
        "torch.cuda.graph" in lower
        or "--enable-cuda-graph" in lower
        or "cuda" in lower
    ), s
    # The canonical CLI constant must appear inside the rendered string.
    assert PLANNED_CAPTURE_CLI in s


# ---------------------------------------------------------------------------
# 6. End-to-end CLI driver
# ---------------------------------------------------------------------------


def _write_problems_jsonl(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"problem": f"Q{i}: {i} + {i} = ?", "answer": str(2 * i)}) + "\n")


def test_run_measure_hybrid_kv_cli_with_dummy_backbone(tmp_path: Path) -> None:
    """Invoke ``scripts/measure_hybrid_kv.py`` end-to-end on a tiny
    synthetic problems file and assert both the Markdown report and
    the JSONL trace are written. We use ``--limit 2 --seeds 0`` so the
    run completes in well under a minute on CPU.
    """
    problems_path = tmp_path / "problems.jsonl"
    _write_problems_jsonl(problems_path, n=2)
    md_path = tmp_path / "out" / "measurement.md"
    jsonl_path = tmp_path / "out" / "measurement.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "measure_hybrid_kv.py"),
            "--problems", str(problems_path),
            "--limit", "2",
            "--seeds", "0",
            "--out", str(md_path),
            "--jsonl", str(jsonl_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert md_path.is_file(), proc.stderr
    assert jsonl_path.is_file(), proc.stderr

    md_text = md_path.read_text(encoding="utf-8")
    assert "KV-reuse hit path NOT YET measured" in md_text
    assert "hybrid_off" in md_text
    assert "hybrid_decision_only" in md_text

    # JSONL trace: 2 problems × 1 seed × 2 modes = 4 rows.
    rows = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4, rows
    assert {r["mode"] for r in rows} == set(ALL_MODES)
    assert {r["problem_id"] for r in rows} == {0, 1}
    # Decision-only rows must have at least one decision call recorded
    # so the §7.7 plumbing is exercised end-to-end via the CLI.
    on_rows = [r for r in rows if r["mode"] == MODE_DECISION_ONLY]
    assert all(r["decision_calls"] >= 1 for r in on_rows), on_rows
