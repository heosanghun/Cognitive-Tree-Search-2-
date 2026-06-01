"""Regression tests for ``scripts/make_anonymous_submission.py``.

These tests guard against silent regressions in the double-blind ZIP
contents. They are CPU-only, do not call any network, and execute in
under a second.

Each scenario corresponds to a real bug we caught (or want to prevent):

* ``logs/`` was once shipped inside the ZIP, exposing tqdm progress
  bars containing torch/cu118 fingerprints and Windows redirect notes
  (audit verdict was PASS because the *content* did not match the
  identity-leak regexes, but the *directory* itself is author-facing).
* ``artifacts/`` and ``data/`` are large + license-restricted; they
  must never be packaged.
* ``sync_to_github.py`` is an author-side maintainer tool that
  references the upstream remote and must not appear in a reviewer
  artefact.
* The ``_audit_anon_zip.py`` script *itself* is excluded because the
  identity-leak regex patterns it carries would, if shipped, defeat
  the purpose of the audit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    """Import scripts/make_anonymous_submission as a fresh module so
    that ``sys.path`` shenanigans elsewhere do not affect it."""
    spec = importlib.util.spec_from_file_location(
        "make_anonymous_submission",
        ROOT / "scripts" / "make_anonymous_submission.py",
    )
    assert spec and spec.loader, "spec or loader is None"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_anonymous_submission"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- 1. EXCLUDE_TOP allowlist completeness --------------------------


REQUIRED_TOP_EXCLUSIONS = (
    # Author-facing dev artefacts
    "logs",
    # Local trained weights (large + license-restricted)
    "artifacts",
    # License-restricted scraped benchmark data
    "data",
    # Hugging Face model cache
    ".hf_cache",
    "gemma-4-E4B",
    "gemma-4-E4B-it",
    # Documentation source (paper PDF, internal notes)
    "doc",
    # Terminal capture for the IDE
    "terminals",
    # Git tooling
    ".git",
    ".github",
    # Editor caches
    ".vscode", ".cursor",
    # Python caches
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    # Virtualenvs
    "venv", ".venv", "env",
    # Pre-anonymize backup of git history
    ".git_backup_pre_anonymize",
)


def test_exclude_top_covers_all_dangerous_directories():
    mod = _load_module()
    missing = [d for d in REQUIRED_TOP_EXCLUSIONS if d not in mod.EXCLUDE_TOP]
    assert not missing, (
        f"EXCLUDE_TOP missing dangerous directories: {missing}. "
        f"Current EXCLUDE_TOP={sorted(mod.EXCLUDE_TOP)}"
    )


# ---------- 2. is_excluded() per-path semantics -----------------------------


@pytest.mark.parametrize(
    "rel_path,should_exclude,reason",
    [
        ("logs/stage1.log", True, "logs/ is author-facing"),
        ("logs/watcher.err", True, "log err files inside logs/"),
        ("artifacts/stage2_meta_value.pt", True, "trained weights"),
        ("data/aime/test_aime_90.jsonl", True, "license-restricted data"),
        (".hf_cache/models--google--gemma-4-E4B/foo.safetensors", True, "model cache"),
        ("doc/Cognitive_Tree_Search.pdf", True, "paper PDF"),
        ("terminals/3.txt", True, "IDE terminal capture"),
        ("__pycache__/foo.cpython-313.pyc", True, "bytecode"),
        ("scripts/_audit_anon_zip.py", True, "audit script (carries leak regexes)"),
        ("scripts/sync_to_github.py", True, "author-facing maintainer tool"),
        ("SUBMISSION_GUIDE_D12.md", True, "author-facing planning doc"),
        ("EXPERIMENTAL_RESULTS.md", True, "author-facing triage doc"),
        # ---- now the *should be included* side ----
        ("README.md", False, "main reviewer-facing readme"),
        ("REVIEWER_FAQ.md", False, "FAQ for reviewers"),
        ("REPRODUCIBILITY.md", False, "reproducibility checklist"),
        ("CHANGELOG.md", False, "changelog"),
        ("LICENSE", False, "license file"),
        ("scripts/run_post_stage2_pipeline.py", False, "main eval pipeline"),
        ("scripts/run_cts_eval_full.py", False, "Table 2 driver"),
        ("scripts/download_all_benchmarks.py", False, "benchmark downloader"),
        ("scripts/make_anonymous_submission.py", False, "submission script itself"),
        ("cts/__init__.py", False, "core library"),
        ("cts/mcts/cts_episode.py", False, "MCTS module"),
        ("tests/test_post_stage2_pipeline.py", False, "regression tests"),
        ("results/sweep_K/sweep_K_plan.txt", False, "small sweep manifest"),
    ],
)
def test_is_excluded_matrix(rel_path, should_exclude, reason):
    mod = _load_module()
    actual = mod.is_excluded(Path(rel_path))
    assert actual == should_exclude, (
        f"is_excluded({rel_path!r}) returned {actual}, expected "
        f"{should_exclude} ({reason})"
    )


# ---------- 3. RESULTS subdir whitelist ------------------------------------


def test_results_whitelist_keeps_summaries_drops_unknown_extensions():
    """``is_excluded`` is the *path*-level filter. Small summaries
    (\\*.md, \\*.json, \\*.txt) under ``results/`` survive; binary blobs
    (\\*.npy, \\*.pt, \\*.bin) are dropped at the path layer regardless
    of size."""
    mod = _load_module()
    # binary / opaque artefacts: dropped by extension whitelist
    assert mod.is_excluded(Path("results/post_stage2_D11/nu_traces/seed0.npy")) is True
    assert mod.is_excluded(Path("results/runs/policy.pt")) is True
    # small reviewer-facing summaries: kept
    assert mod.is_excluded(Path("results/sweep_K/sweep_K_summary.md")) is False
    assert mod.is_excluded(Path("results/contamination/aime_screen_90.md")) is False
    assert mod.is_excluded(Path("results/sweep_K/sweep_K_plan.txt")) is False


def test_iter_files_size_caps_results_outputs(tmp_path: Path):
    """Even when a per-seed/per-method ``results/.../*.json`` slips
    through the path filter, the size cap (``RESULTS_FILE_MAX_BYTES``)
    must drop oversized raw outputs from the ZIP. This protects against
    a bulky 5-MB Table 17 json file accidentally shipping.

    The fixture lives under ``results/sweep_K`` (a reviewer-relevant
    sweep manifest directory that is NOT path-excluded) so the size cap
    is the only thing protecting the ZIP from bulky raw dumps.
    """
    mod = _load_module()
    cap = mod.RESULTS_FILE_MAX_BYTES
    fake_root = tmp_path / "fake_repo"
    (fake_root / "results" / "sweep_K").mkdir(parents=True)
    small = fake_root / "results" / "sweep_K" / "summary.json"
    small.write_text('{"acc": 0.567}', encoding="utf-8")
    big = fake_root / "results" / "sweep_K" / "raw_dump.json"
    # +1 to be definitively over the cap
    big.write_bytes(b"x" * (cap + 1))

    files = list(mod.iter_files(fake_root))
    rels = {f.relative_to(fake_root).as_posix() for f in files}
    assert "results/sweep_K/summary.json" in rels
    assert "results/sweep_K/raw_dump.json" not in rels, (
        "size cap failed: bulky raw outputs leaked into the ZIP"
    )
