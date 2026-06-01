"""Tests for paper Table 19 ν cross-domain statistics aggregator.

Covers:
  * end-to-end aggregator + summariser + renderer on a synthetic
    2-method × 3-domain × 2-seed × 4-problem fixture;
  * directional ``↑`` marker fires on a synthesized
    ``nu_expl_AIME > nu_expl_GSM8K`` significance case;
  * empty / missing ``nu_trace`` returns an empty DataFrame and the
    renderer writes a "no data" banner (no crash);
  * round-trip integration through the actual ``cts/mcts/cts_episode.py``
    so the ``nu_trace`` list-passing hook is proven to populate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from cts.eval.nu_stats import (
    DEFAULT_DOMAIN_MAP,
    NU_COMPONENTS,
    aggregate_nu_traces,
    render_table19_markdown,
    summarize_table19,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_nu_jsonl(
    path: Path,
    *,
    method: str,
    benchmark: str,
    seed: int,
    problems: int,
    nu_per_step: Dict[str, List[List[float]]],
) -> None:
    """Write a JSONL file with `problems` lines, each carrying a per-step
    ν trace. ``nu_per_step[component]`` must have shape ``[problems][steps]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pi in range(problems):
            rec = {
                "method": method,
                "benchmark": benchmark,
                "seed": seed,
                "problem_id": f"{benchmark}/{pi}",
                "nu_trace": {
                    comp: nu_per_step[comp][pi] for comp in NU_COMPONENTS
                },
            }
            f.write(json.dumps(rec) + "\n")


def _const_trace(value: float, problems: int, steps: int) -> List[List[float]]:
    return [[value] * steps for _ in range(problems)]


# ---------------------------------------------------------------------------
# Test 1 — end-to-end on a synthetic 2-method × 3-domain × 2-seed × 4-problem fixture
# ---------------------------------------------------------------------------

def test_aggregate_summarize_render_end_to_end(tmp_path: Path) -> None:
    """Synthesise the full Table 19 fixture and round-trip through every
    public function. Asserts the schema of every output, not the values."""
    methods = ["cts_4nu", "cts_2nu"]
    benchmark_to_constant = {
        # math domain
        "aime":     {"nu_expl": 1.40, "nu_tol": 0.50, "nu_temp": 1.10, "nu_act": 0.60},
        "gsm8k":    {"nu_expl": 0.80, "nu_tol": 0.50, "nu_temp": 1.00, "nu_act": 0.95},
        "math500":  {"nu_expl": 1.05, "nu_tol": 0.50, "nu_temp": 1.05, "nu_act": 0.78},
        # code domain
        "humaneval":{"nu_expl": 0.95, "nu_tol": 0.50, "nu_temp": 1.20, "nu_act": 0.85},
        # reasoning domain
        "arc_agi_text": {"nu_expl": 0.90, "nu_tol": 0.50, "nu_temp": 1.00, "nu_act": 0.90},
    }
    seeds = [0, 1]
    problems = 4
    steps = 6

    paths: List[Path] = []
    for method in methods:
        for bench, constants in benchmark_to_constant.items():
            for seed in seeds:
                p = tmp_path / "runs" / f"{method}_{bench}_seed{seed}.jsonl"
                _write_nu_jsonl(
                    p, method=method, benchmark=bench, seed=seed,
                    problems=problems,
                    nu_per_step={
                        comp: _const_trace(constants[comp], problems, steps)
                        for comp in NU_COMPONENTS
                    },
                )
                paths.append(p)

    df = aggregate_nu_traces(paths, DEFAULT_DOMAIN_MAP)

    expected_cols = {
        "method", "benchmark", "domain", "problem_id",
        "seed", "nu_component", "nu_value",
    }
    assert expected_cols.issubset(set(df.columns))
    # 2 methods × 5 benchmarks × 2 seeds × 4 problems × 4 components × 6 steps
    assert len(df) == 2 * 5 * 2 * 4 * 4 * 6

    summary = summarize_table19(df)
    expected_summary_cols = {
        "method", "domain",
        "nu_expl_mean_std", "nu_tol_mean_std",
        "nu_temp_mean_std", "nu_act_mean_std",
        "p_nu_expl_aime_gt_gsm8k", "p_nu_act_gsm8k_gt_aime",
        "marker_nu_expl", "marker_nu_act",
    }
    assert expected_summary_cols.issubset(set(summary.columns))
    # 2 methods × 3 domains
    assert len(summary) == 2 * 3
    # All ν cells filled out for every (method, domain).
    for col in ("nu_expl_mean_std", "nu_tol_mean_std",
                "nu_temp_mean_std", "nu_act_mean_std"):
        assert summary[col].astype(bool).all(), f"missing values in {col}"

    out_path = tmp_path / "table19" / "nu_stats.md"
    render_table19_markdown(summary, out_path)
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "Table 19" in text
    assert "cts_4nu" in text and "cts_2nu" in text
    assert "math" in text and "code" in text and "reasoning" in text


# ---------------------------------------------------------------------------
# Test 2 — directional ↑ marker fires on a clear AIME > GSM8K case
# ---------------------------------------------------------------------------

def test_summarize_marks_paper_directional_claims(tmp_path: Path) -> None:
    """Build a fixture where ``nu_expl(AIME) >> nu_expl(GSM8K)`` (Welch p
    will be tiny) and assert (a) p_corr drops below 0.05, (b) the rendered
    Markdown contains the ``↑`` marker on the math row."""
    method = "cts_4nu"
    seeds = [0, 1, 2]
    problems = 6
    steps = 8
    paths: List[Path] = []

    # AIME: high nu_expl with small noise; GSM8K: low nu_expl. Same shape for
    # the other components so the comparison is uncontaminated.
    def _scaled_trace(base: float, jitter: float, n_p: int, n_s: int) -> List[List[float]]:
        out: List[List[float]] = []
        for pi in range(n_p):
            row = [base + jitter * ((pi + s) % 5 - 2) / 10.0 for s in range(n_s)]
            out.append(row)
        return out

    for bench, expl_base in [("aime", 1.50), ("gsm8k", 0.60), ("math500", 1.00)]:
        for seed in seeds:
            p = tmp_path / "runs" / f"{method}_{bench}_seed{seed}.jsonl"
            # nu_act intentionally swapped so we ALSO trigger the second
            # paper-highlighted claim (nu_act_GSM8K > nu_act_AIME).
            act_base = 0.95 if bench == "gsm8k" else (0.55 if bench == "aime" else 0.78)
            _write_nu_jsonl(
                p, method=method, benchmark=bench, seed=seed,
                problems=problems,
                nu_per_step={
                    "nu_expl": _scaled_trace(expl_base, 0.05, problems, steps),
                    "nu_tol":  _const_trace(0.50, problems, steps),
                    "nu_temp": _const_trace(1.00, problems, steps),
                    "nu_act":  _scaled_trace(act_base, 0.04, problems, steps),
                },
            )
            paths.append(p)

    df = aggregate_nu_traces(paths)
    summary = summarize_table19(df)

    math_row = summary[(summary["method"] == method) & (summary["domain"] == "math")]
    assert len(math_row) == 1
    p_expl_corr = float(math_row["p_nu_expl_aime_gt_gsm8k_corr"].iloc[0])
    p_act_corr = float(math_row["p_nu_act_gsm8k_gt_aime_corr"].iloc[0])
    assert p_expl_corr < 0.05, (
        f"directional claim nu_expl(AIME)>nu_expl(GSM8K) should be significant; "
        f"corr_p={p_expl_corr}"
    )
    assert p_act_corr < 0.05, (
        f"directional claim nu_act(GSM8K)>nu_act(AIME) should be significant; "
        f"corr_p={p_act_corr}"
    )

    out_path = tmp_path / "out.md"
    render_table19_markdown(summary, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "↑" in text, "expected paper-style up-arrow marker in rendered table"


# ---------------------------------------------------------------------------
# Test 3 — empty / missing nu_trace produces empty df + banner output
# ---------------------------------------------------------------------------

def test_aggregator_empty_when_no_nu_trace(tmp_path: Path) -> None:
    """A JSONL file whose lines lack `nu_trace` (or that doesn't exist at
    all) must yield an empty DataFrame, and the renderer must write a
    clear "no data" banner."""
    # Case A: file exists but every line lacks `nu_trace`.
    no_trace = tmp_path / "no_trace.jsonl"
    with open(no_trace, "w", encoding="utf-8") as f:
        f.write(json.dumps({"method": "cts_4nu", "benchmark": "aime", "seed": 0,
                            "problem_id": "aime/0"}) + "\n")
        f.write(json.dumps({"method": "cts_4nu", "benchmark": "aime", "seed": 0,
                            "problem_id": "aime/1"}) + "\n")

    # Case B: file does not exist at all.
    missing = tmp_path / "does_not_exist.jsonl"

    df = aggregate_nu_traces([no_trace, missing], DEFAULT_DOMAIN_MAP)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    # Schema is preserved even when empty so downstream consumers can still
    # safely call `.columns`.
    assert "nu_value" in df.columns
    assert "method" in df.columns

    summary = summarize_table19(df)
    assert isinstance(summary, pd.DataFrame)
    assert summary.empty

    out_path = tmp_path / "empty.md"
    render_table19_markdown(summary, out_path)
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "No `nu_trace` data" in text or "no data" in text.lower()
    # Banner must mention how to fix it (--nu-trace-dir hint).
    assert "nu-trace-dir" in text or "CTS_NU_TRACE_DIR" in text


def test_aggregator_skips_unknown_benchmark(tmp_path: Path) -> None:
    """Lines for benchmarks not in the domain map must be silently dropped,
    not raise — this lets reviewers mix paper and non-paper runs freely."""
    p = tmp_path / "mixed.jsonl"
    _write_nu_jsonl(
        p, method="cts_4nu", benchmark="unknown_bench", seed=0, problems=2,
        nu_per_step={comp: _const_trace(1.0, 2, 3) for comp in NU_COMPONENTS},
    )
    df = aggregate_nu_traces([p], {"aime": "math"})
    assert df.empty


# ---------------------------------------------------------------------------
# Test 4 — round-trip through cts_full_episode (real integration)
# ---------------------------------------------------------------------------

def test_cts_full_episode_populates_nu_trace_list_when_provided() -> None:
    """The whole reason for the persistence hook: passing a `nu_trace` list
    to `cts_full_episode` must result in NuVectors actually being appended.
    Mirrors the `tests/test_cts_full_episode.py` mock-backbone pattern so
    we don't need any GPU / real Gemma weights."""
    import torch

    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.critic.neuro_critic import NeuroCritic
    from cts.mcts.cts_episode import cts_full_episode
    from cts.policy.meta_policy import MetaPolicy
    from cts.types import NuVector

    torch.manual_seed(2026)
    d, W, K = 16, 2, 4
    bb = MockTinyBackbone(hidden=d, num_layers=4)
    meta = MetaPolicy(text_dim=d, hidden=32, W=W)
    critic = NeuroCritic(z_dim=d)

    nu_trace: List[NuVector] = []
    result = cts_full_episode(
        "Q: round-trip test",
        backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K,
        tau_budget=5e12,
        broyden_max_iter=3, broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=10.0,
        z0_seed=11, selection_seed=11,
        nu_config_mode="4nu",
        nu_trace=nu_trace,
    )
    assert result is not None
    assert nu_trace, "nu_trace list was provided but never populated"
    for nv in nu_trace:
        assert isinstance(nv, NuVector)
        for comp in NU_COMPONENTS:
            v = float(getattr(nv, comp))
            assert not (v != v), f"NaN in {comp}"  # NaN check
            assert v > 0.0, f"unexpected non-positive {comp}={v}"


def test_cts_full_episode_default_does_not_capture_trace() -> None:
    """Backward-compat: not passing ``nu_trace`` must keep zero overhead and
    cannot raise. This guards the existing 308-test suite from regressing."""
    import torch

    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.critic.neuro_critic import NeuroCritic
    from cts.mcts.cts_episode import cts_full_episode
    from cts.policy.meta_policy import MetaPolicy

    torch.manual_seed(2026)
    d, W, K = 16, 2, 4
    bb = MockTinyBackbone(hidden=d, num_layers=4)
    meta = MetaPolicy(text_dim=d, hidden=32, W=W)
    critic = NeuroCritic(z_dim=d)

    res = cts_full_episode(
        "Q: default trace off",
        backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K,
        tau_budget=5e12,
        broyden_max_iter=3, broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=10.0,
        z0_seed=11, selection_seed=11,
    )
    assert res is not None


# ---------------------------------------------------------------------------
# Test 5 — round-trip the JSONL persistence via the dispatcher helper
# ---------------------------------------------------------------------------

def test_run_cts_eval_full_helpers_round_trip(tmp_path: Path) -> None:
    """Drive ``_append_nu_trace_record`` directly (bypassing the GPU
    dispatcher) and confirm the resulting JSONL is consumable by the
    aggregator. This pins the file-format contract between
    ``scripts/run_cts_eval_full.py`` and ``cts/eval/nu_stats.py``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_run_cts_eval_full",
        Path(__file__).resolve().parent.parent / "scripts" / "run_cts_eval_full.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from cts.types import NuVector

    nu_buf = [
        NuVector(nu_expl=1.2, nu_tol=0.5, nu_temp=1.0, nu_act=0.6),
        NuVector(nu_expl=1.4, nu_tol=0.5, nu_temp=1.1, nu_act=0.6),
    ]
    out_path = tmp_path / "cts_4nu_aime_seed0.jsonl"
    mod._append_nu_trace_record(  # type: ignore[attr-defined]
        out_path, method="cts_4nu", benchmark="aime", seed=0,
        problem_id="aime/0", nu_buf=nu_buf,
    )
    mod._append_nu_trace_record(  # type: ignore[attr-defined]
        out_path, method="cts_4nu", benchmark="aime", seed=0,
        problem_id="aime/1", nu_buf=nu_buf,
    )

    df = aggregate_nu_traces([out_path])
    assert not df.empty
    assert set(df["method"].unique()) == {"cts_4nu"}
    assert set(df["benchmark"].unique()) == {"aime"}
    assert set(df["domain"].unique()) == {"math"}
    # 2 problems × 4 components × 2 steps
    assert len(df) == 2 * 4 * 2
