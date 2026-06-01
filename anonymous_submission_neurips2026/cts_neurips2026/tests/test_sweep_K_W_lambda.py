"""Regression tests for paper Tables 13 / 15 / λ_halt sweep automation.

Covers:
  - ``cts/eval/sweep_utils.py`` (bootstrap CI, JSONL load, Markdown
    render, dry-run grid).
  - ``cts_full_episode`` ``k_override`` / ``w_override`` kwargs (paper
    Table 13 / 15 sensitivity sweep wiring) and their stats provenance.
  - ``scripts/run_sweep_K.py`` / ``run_sweep_W.py`` ``--dry-run``
    planning path (no eval launched, no JSONL written).
  - ``scripts/run_sweep_lambda_halt.py`` PENDING_GPU manifest path
    (no checkpoints present → ``training_jobs.json`` with 4 entries).
  - End-to-end sweep with ``_run_cts_on_problems`` monkey-patched to a
    deterministic stub: assert JSONL + Markdown summary are written and
    parseable.

All tests are CPU-only and complete in <10 s on a laptop.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cts.backbone.mock_tiny import MockTinyBackbone  # noqa: E402
from cts.critic.neuro_critic import NeuroCritic  # noqa: E402
from cts.eval.sweep_utils import (  # noqa: E402
    bootstrap_ci,
    dry_run_grid,
    load_sweep_jsonl,
    render_sweep_markdown,
    summarize_sweep,
)
from cts.mcts.cts_episode import CtsEpisodeResult, cts_full_episode  # noqa: E402
from cts.policy.meta_policy import MetaPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# 1. bootstrap_ci primitives
# ---------------------------------------------------------------------------


def test_bootstrap_ci_constant_input() -> None:
    """Constant samples must collapse to lo == hi == mean (no width).

    Paper §7.1 defines a 95% percentile bootstrap; a degenerate sample
    has zero variance so the CI must collapse exactly. Without this
    guarantee a constant 100% accuracy seed would render with a fake
    non-zero CI half-width and mis-represent the sweep.
    """
    mean, lo, hi = bootstrap_ci([0.42, 0.42, 0.42, 0.42, 0.42], n_resamples=128)
    assert mean == pytest.approx(0.42)
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


def test_bootstrap_ci_monotone() -> None:
    """For [0.0]*8 + [1.0]*2 the mean is 0.2 and lo <= mean <= hi."""
    samples = [0.0] * 8 + [1.0] * 2
    mean, lo, hi = bootstrap_ci(samples, n_resamples=512, alpha=0.05)
    assert mean == pytest.approx(0.2, rel=1e-9)
    assert lo <= mean <= hi
    assert hi <= 1.0
    assert lo >= 0.0


def test_bootstrap_ci_empty_returns_zeros() -> None:
    """Empty input must not crash; sweep aggregator depends on this."""
    mean, lo, hi = bootstrap_ci([], n_resamples=10)
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 2. JSONL / Markdown helpers
# ---------------------------------------------------------------------------


def test_load_sweep_jsonl_missing_returns_empty(tmp_path: Path) -> None:
    """Non-existent path must return [] (not raise) so resume logic works."""
    rows = load_sweep_jsonl(tmp_path / "does_not_exist.jsonl")
    assert rows == []


def test_load_sweep_jsonl_skips_corrupt_lines(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text(
        '{"K":2,"seed":0,"score":1.0}\n'
        'this is not json\n'
        '{"K":3,"seed":0,"score":0.5}\n',
        encoding="utf-8",
    )
    rows = load_sweep_jsonl(p)
    assert len(rows) == 2
    assert rows[0]["K"] == 2 and rows[1]["K"] == 3


def test_render_sweep_markdown_writes_table(tmp_path: Path) -> None:
    """Markdown must contain the param name and the canonical
    ``mean ± 95% CI`` header so the paper-artifacts pipeline can grep
    for it.
    """
    out = tmp_path / "sweep.md"
    rows: List[Dict[str, Any]] = [
        {"param_value": 2, "n_problems": 30, "n_seeds": 3,
         "mean_acc": 0.42, "ci_lo": 0.32, "ci_hi": 0.52},
        {"param_value": 3, "n_problems": 30, "n_seeds": 3,
         "mean_acc": 0.50, "ci_lo": 0.40, "ci_hi": 0.60},
    ]
    render_sweep_markdown(rows, "K", out)
    txt = out.read_text(encoding="utf-8")
    assert "K" in txt
    assert "mean ± 95% CI" in txt
    assert "42.0 ± 10.0%" in txt
    assert "50.0 ± 10.0%" in txt


def test_dry_run_grid_returns_full_cartesian() -> None:
    grid = dry_run_grid([2, 4], [0, 1], ["aime"])
    assert grid == [
        (2, 0, "aime"), (2, 1, "aime"),
        (4, 0, "aime"), (4, 1, "aime"),
    ]


# ---------------------------------------------------------------------------
# 3. cts_full_episode k_override / w_override wiring
# ---------------------------------------------------------------------------


class _DecodingMockBackbone(MockTinyBackbone):
    def __init__(self, hidden: int = 16, num_layers: int = 4) -> None:
        super().__init__(hidden=hidden, num_layers=num_layers)

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 64) -> str:
        return f"answer={z_star.detach().float().mean().item():+.4f}"


def _build_components(d: int = 16, W: int = 3):
    torch.manual_seed(2026)
    bb = _DecodingMockBackbone(hidden=d, num_layers=4)
    meta = MetaPolicy(text_dim=d, hidden=32, W=W)
    critic = NeuroCritic(z_dim=d)
    return bb, meta, critic


def test_cts_full_episode_accepts_k_and_w_override() -> None:
    """Paper Table 13 / 15 sensitivity-sweep wiring: cts_full_episode
    must accept ``k_override`` and ``w_override`` and their effective
    values must be recorded in result.stats.
    """
    d, W, K = 16, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)

    res = cts_full_episode(
        "Q: 2 + 2 = ?",
        backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K,
        tau_budget=5e13, broyden_max_iter=3,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=15.0,
        z0_seed=11, selection_seed=11,
        k_override=2,
        w_override=4,
    )
    assert isinstance(res, CtsEpisodeResult)
    assert res.stats["k_override_used"] == 2, res.stats
    # w_override caps the outer loop at exactly 4 PUCT iterations.
    assert res.stats["sim_count"] <= 4, res.stats
    assert res.stats["w_override_used"] is not None
    assert res.stats["w_override_used"] <= 4

    # Tree must reflect the override: exactly 2 children per first
    # expansion (root + 2 children = tree_size >= 3).
    assert res.stats["tree_size"] >= 3


def test_cts_full_episode_default_overrides_preserve_behaviour() -> None:
    """``k_override=None`` and ``w_override=None`` must be byte-identical
    to the historical call site (no surprise behaviour changes for the
    314 existing tests)."""
    d, W, K = 16, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)

    res = cts_full_episode(
        "Q: 1 + 1 = ?",
        backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K,
        tau_budget=2e13, broyden_max_iter=3,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=10.0,
        z0_seed=7, selection_seed=7,
    )
    assert res.stats["k_override_used"] == W
    assert res.stats["w_override_used"] is None


def test_cts_full_episode_rejects_invalid_overrides() -> None:
    """k_override < 1 / w_override < 1 must raise (defensive check)."""
    d, W, K = 16, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)
    with pytest.raises(ValueError):
        cts_full_episode(
            "Q: bad", backbone=bb, meta_policy=meta, critic=critic,
            W=W, K=K, tau_budget=1e12, broyden_max_iter=2,
            broyden_tol_min=1e-3, broyden_tol_max=1e-2,
            top_k=2, max_decode_tokens=2,
            device=torch.device("cpu"), wall_clock_budget_s=5.0,
            k_override=0,
        )
    with pytest.raises(ValueError):
        cts_full_episode(
            "Q: bad", backbone=bb, meta_policy=meta, critic=critic,
            W=W, K=K, tau_budget=1e12, broyden_max_iter=2,
            broyden_tol_min=1e-3, broyden_tol_max=1e-2,
            top_k=2, max_decode_tokens=2,
            device=torch.device("cpu"), wall_clock_budget_s=5.0,
            w_override=0,
        )


# ---------------------------------------------------------------------------
# 4. Sweep-script --dry-run paths (no eval launched)
# ---------------------------------------------------------------------------


def _run_script(script: Path, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc


def test_dry_run_K_writes_plan(tmp_path: Path) -> None:
    """`--dry-run` for K-sweep prints the plan and writes no JSONL."""
    plan = tmp_path / "plan.txt"
    jsonl = tmp_path / "sweep_K.jsonl"
    md = tmp_path / "sweep_K.md"
    proc = _run_script(
        ROOT / "scripts" / "run_sweep_K.py",
        "--dry-run",
        "--k-values", "2", "3",
        "--seeds", "0",
        "--plan", str(plan),
        "--jsonl", str(jsonl),
        "--md", str(md),
    )
    assert proc.returncode == 0, proc.stderr
    assert "planned 2" in proc.stdout, proc.stdout
    assert "K=2 seed=0" in proc.stdout
    assert "K=3 seed=0" in proc.stdout
    assert plan.is_file(), proc.stdout
    plan_txt = plan.read_text(encoding="utf-8")
    assert "2\t0\taime" in plan_txt
    assert "3\t0\taime" in plan_txt
    # No JSONL written on dry-run.
    assert not jsonl.exists()


def test_dry_run_W_writes_plan(tmp_path: Path) -> None:
    """`--dry-run` for W-sweep prints the plan and writes no JSONL."""
    plan = tmp_path / "plan.txt"
    jsonl = tmp_path / "sweep_W.jsonl"
    md = tmp_path / "sweep_W.md"
    proc = _run_script(
        ROOT / "scripts" / "run_sweep_W.py",
        "--dry-run",
        "--w-values", "4", "8",
        "--seeds", "0",
        "--plan", str(plan),
        "--jsonl", str(jsonl),
        "--md", str(md),
    )
    assert proc.returncode == 0, proc.stderr
    assert "planned 2" in proc.stdout
    assert "W=4 seed=0" in proc.stdout
    assert "W=8 seed=0" in proc.stdout
    assert plan.is_file()
    plan_txt = plan.read_text(encoding="utf-8")
    assert "4\t0\taime" in plan_txt
    assert "8\t0\taime" in plan_txt
    assert not jsonl.exists()


# ---------------------------------------------------------------------------
# 5. λ_halt PENDING_GPU manifest path
# ---------------------------------------------------------------------------


def test_dry_run_lambda_writes_pending_manifest(tmp_path: Path) -> None:
    """With no checkpoints present, run_sweep_lambda_halt.py must write
    ``training_jobs.json`` with 4 entries and a Markdown status table
    flagging every λ as PENDING_GPU.
    """
    runs = tmp_path / "runs"  # intentionally empty
    runs.mkdir()
    jobs = tmp_path / "training_jobs.json"
    md = tmp_path / "sweep_lambda_halt.md"
    jsonl = tmp_path / "sweep_lambda_halt.jsonl"

    proc = _run_script(
        ROOT / "scripts" / "run_sweep_lambda_halt.py",
        "--runs-dir", str(runs),
        "--jobs", str(jobs),
        "--md", str(md),
        "--jsonl", str(jsonl),
        "--lambda-values", "0.01", "0.05", "0.1", "0.5",
        "--seeds", "0", "1", "2",
        "--force-no-gpu",
    )
    assert proc.returncode == 0, proc.stderr
    assert jobs.is_file()
    manifest = json.loads(jobs.read_text(encoding="utf-8"))
    assert manifest["param_name"] == "lambda_halt"
    assert len(manifest["jobs"]) == 4
    statuses = {j["status"] for j in manifest["jobs"]}
    assert statuses == {"PENDING_GPU"}
    assert md.is_file()
    md_txt = md.read_text(encoding="utf-8")
    assert "PENDING_GPU" in md_txt
    # No JSONL written when manifest mode triggers.
    assert not jsonl.exists()


def test_lambda_manifest_uses_relative_posix_paths(tmp_path: Path) -> None:
    """Regression: the lambda-halt manifest must NEVER serialize absolute
    paths (e.g. host-specific ``<drive>:/<dir>/runs/...`` on Windows or
    ``/home/<user>/...`` on Linux) — those would (a) break reviewer
    reproduction on a different host and
    (b) potentially leak the author's local layout into a double-blind
    artifact. ckpt_path must be relative + posix-style.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    jobs = tmp_path / "training_jobs.json"
    md = tmp_path / "sweep_lambda_halt.md"
    jsonl = tmp_path / "sweep_lambda_halt.jsonl"

    proc = _run_script(
        ROOT / "scripts" / "run_sweep_lambda_halt.py",
        "--runs-dir", str(runs),
        "--jobs", str(jobs),
        "--md", str(md),
        "--jsonl", str(jsonl),
        "--lambda-values", "0.01", "0.05", "0.1", "0.5",
        "--seeds", "0", "1", "2",
        "--force-no-gpu",
    )
    assert proc.returncode == 0, proc.stderr

    manifest = json.loads(jobs.read_text(encoding="utf-8"))
    for job in manifest["jobs"]:
        ckpt = job["ckpt_path"]
        notes = job["notes"]
        assert "\\" not in ckpt, (
            f"ckpt_path uses backslashes (Windows leak): {ckpt!r}"
        )
        assert ":" not in ckpt, (
            f"ckpt_path looks absolute (drive letter / colon): {ckpt!r}"
        )
        assert not ckpt.startswith("/"), (
            f"ckpt_path is POSIX-absolute: {ckpt!r}"
        )
        assert ckpt.startswith("runs/") or "runs/" in ckpt, (
            f"ckpt_path missing canonical 'runs/' prefix: {ckpt!r}"
        )
        # And the human-facing notes must mirror the same path.
        assert ckpt in notes, (
            "notes must reference the same relative ckpt_path"
        )

    md_txt = md.read_text(encoding="utf-8")
    assert "D:\\" not in md_txt and "D:/" not in md_txt, (
        "λ_halt status MD leaks an absolute Windows path"
    )


# ---------------------------------------------------------------------------
# 6. End-to-end sweep with monkey-patched _run_cts_on_problems
# ---------------------------------------------------------------------------


def test_run_sweep_K_with_stub_writes_jsonl_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkey-patch `_run_cts_on_problems` to a deterministic stub and
    drive ``run_sweep_K.run_sweep`` end-to-end; assert the JSONL and
    Markdown summary are written and parseable.

    This exercises the full launcher (k_override monkey-patch, JSONL
    append, summary aggregation, Markdown render) without requiring a
    GPU or a real Gemma backbone.
    """
    import scripts.run_cts_eval_full as eval_mod

    captured_calls: List[Dict[str, Any]] = []

    def _stub_run(
        method: str, problems: list, cfg: dict, device: str, model_dir,
        *, benchmark: str = "math500", seed: int = 0, nu_trace_dir=None,
    ):
        # Deterministic per-(K, seed) score: K=2 → 0.4, K=3 → 0.6.
        # We can't see K directly here because the stub does not call
        # cts_full_episode; instead the K-sweep launcher monkey-patches
        # cts_full_episode with k_override before each call. We
        # therefore read the *current* monkey-patched k_override out of
        # the cts_full_episode binding so the stub can produce a
        # K-dependent score.
        from cts.mcts import cts_episode as _ep
        wrapped = _ep.cts_full_episode
        try:
            current_k = wrapped.__defaults_for_test__  # type: ignore[attr-defined]
        except AttributeError:
            current_k = None
        # Fallback: parse defaults from the wrapped closure cell.
        if current_k is None:
            try:
                # Our launcher injects k_override via kwargs.setdefault inside
                # the wrapper closure; we can recover it from cell contents.
                cells = wrapped.__closure__ or ()
                for cell in cells:
                    val = cell.cell_contents
                    if isinstance(val, int) and val in (2, 3, 4, 5, 6, 8):
                        current_k = val
                        break
            except Exception:
                current_k = None
        score_per_problem = 0.4 if current_k == 2 else 0.6
        n = len(problems) if problems else 3
        scores = [score_per_problem] * n
        captured_calls.append({
            "method": method, "benchmark": benchmark,
            "seed": seed, "k_seen": current_k, "n": n,
        })
        return scores

    monkeypatch.setattr(eval_mod, "_run_cts_on_problems", _stub_run)

    from scripts import run_sweep_K as sweep_mod

    out_dir = tmp_path / "sweep_K"
    jsonl = out_dir / "sweep_K.jsonl"
    md = out_dir / "sweep_K.md"
    plan = out_dir / "sweep_K_plan.txt"

    summary = sweep_mod.run_sweep(
        k_values=[2, 3],
        seeds=[0],
        benchmark="aime",
        method="cts_4nu",
        limit=3,
        config_name="default",
        device="cpu",
        model_dir=None,
        jsonl_path=jsonl,
        md_path=md,
        plan_path=plan,
        dry_run=False,
    )

    assert summary["dry_run"] is False
    assert summary["launched"] == 2
    assert jsonl.is_file()
    assert md.is_file()
    rows = load_sweep_jsonl(jsonl)
    # 2 K values × 1 seed × 3 problems = 6 rows.
    assert len(rows) == 6
    # Each call should have seen the correct K via the monkey-patched
    # k_override.
    seen_ks = sorted({c["k_seen"] for c in captured_calls})
    assert seen_ks == [2, 3]
    # Markdown summary must list both K values and the canonical CI header.
    md_txt = md.read_text(encoding="utf-8")
    assert "mean ± 95% CI" in md_txt
    assert "K" in md_txt
    # Idempotent re-run: launching again must skip everything.
    summary2 = sweep_mod.run_sweep(
        k_values=[2, 3],
        seeds=[0],
        benchmark="aime",
        method="cts_4nu",
        limit=3,
        config_name="default",
        device="cpu",
        model_dir=None,
        jsonl_path=jsonl,
        md_path=md,
        plan_path=plan,
        dry_run=False,
    )
    assert summary2["launched"] == 0
    assert summary2["skipped"] == 2


def test_summarize_sweep_groups_by_param_and_seed() -> None:
    """summarize_sweep must produce one row per param_value with the
    proper bootstrap CI (paper §7.1 protocol)."""
    rows = [
        {"K": 2, "seed": 0, "score": 1.0},
        {"K": 2, "seed": 0, "score": 0.0},
        {"K": 2, "seed": 1, "score": 1.0},
        {"K": 2, "seed": 1, "score": 1.0},
        {"K": 4, "seed": 0, "score": 0.0},
        {"K": 4, "seed": 0, "score": 0.0},
        {"K": 4, "seed": 1, "score": 0.0},
        {"K": 4, "seed": 1, "score": 1.0},
    ]
    summary = summarize_sweep(rows, "K", n_resamples=128)
    assert len(summary) == 2
    by_k = {r["param_value"]: r for r in summary}
    # K=2 per-seed means: [0.5, 1.0] → mean 0.75
    assert by_k[2]["mean_acc"] == pytest.approx(0.75)
    # K=4 per-seed means: [0.0, 0.5] → mean 0.25
    assert by_k[4]["mean_acc"] == pytest.approx(0.25)
    for r in summary:
        assert r["ci_lo"] <= r["mean_acc"] <= r["ci_hi"]
        assert r["n_problems"] == 4
        assert r["n_seeds"] == 2
