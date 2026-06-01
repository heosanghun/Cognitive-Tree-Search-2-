"""Regression tests for ``scripts/run_post_stage2_pipeline.py``.

The master post-Stage-2 evaluation pipeline orchestrates paper Tables
2, 17, and 19 plus the anonymous ZIP rebuild. End-to-end execution
requires GPU + the patched Stage 2 checkpoint, so these tests instead
exercise the *control plane*:

* argparser surface (--seeds / --device / --skip-* / --smoke flags).
* phase_verify_stage2 status logic against a synthesized in-memory
  checkpoint (PASS / WARN / FAIL branches).
* run_pipeline driver respects --skip-table2 / --skip-table17 /
  --skip-table19 / --skip-zip and never executes the corresponding
  phase function.
* Phase functions catch (rather than re-raise) sub-process failures so
  one bad seed cannot abort a 30-hour run.

These tests are CPU-only and do not import torch unless phase 1 is
exercised; the synthesized checkpoint test does require torch but uses
a tiny ``OrderedDict`` so it stays under 1 ms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_post_stage2_pipeline as mod  # noqa: E402


# ---------- argparser surface ---------------------------------------------


def test_argparser_default_flags():
    args = mod._build_argparser().parse_args([])
    assert args.seeds == 5
    assert args.device == "cuda:0"
    assert args.output_root == "results/post_stage2"
    assert args.smoke is False
    assert args.skip_verify is False
    assert args.skip_table2 is False
    assert args.skip_table17 is False
    assert args.skip_table19 is False
    assert args.skip_zip is False


def test_argparser_smoke_and_skip_flags():
    args = mod._build_argparser().parse_args([
        "--smoke", "--skip-verify", "--skip-table2", "--skip-table17",
        "--skip-table19", "--skip-zip", "--seeds", "1",
    ])
    assert args.smoke is True
    assert args.skip_verify is True
    assert args.skip_table2 is True
    assert args.skip_table17 is True
    assert args.skip_table19 is True
    assert args.skip_zip is True
    assert args.seeds == 1


def test_run_pipeline_skip_verify_skips_phase_1(tmp_path, monkeypatch):
    """--skip-verify must skip phase 1 (verify_stage2) entirely.

    Production runs MUST keep the phase 1 gate; this flag is only for
    pre-flight wiring tests against pre-patch checkpoints.
    """
    out = tmp_path / "out"
    args = mod._build_argparser().parse_args([
        "--skip-verify", "--skip-table2", "--skip-table17",
        "--skip-table19", "--skip-zip",
        "--output-root", str(out),
    ])
    status = mod.run_pipeline(args)
    assert status["phases"]["verify_stage2"]["status"] == "SKIP"
    # final_verdict should be SKIPPED_ALL (every phase skipped)
    assert status["final_verdict"] == "SKIPPED_ALL"


# ---------- phase 1: verify_stage2 -----------------------------------------


def test_verify_stage2_missing_ckpt_returns_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "FAIL"
    assert "missing" in result["details"]["reason"]


def test_verify_stage2_too_small_returns_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"x")  # 1 byte
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "FAIL"
    assert "small" in result["details"]["reason"]


def _bigtensor(torch_module):
    """Return a tensor large enough that the resulting state-dict beats
    the 0.1 MB sanity threshold in phase_verify_stage2 (which is set
    to that value to detect truncated/corrupt ckpts in the wild)."""
    # 200 x 200 x 4 bytes ~= 160 KB > 0.1 MB threshold per substate.
    return torch_module.zeros(200, 200)


def test_verify_stage2_paper_faithful_meta_returns_pass(tmp_path, monkeypatch):
    """Paper §6.2 P0-4 ckpt has collect_batch=64, ppo_epochs=4."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": {"linear.weight": _bigtensor(torch)},
        "critic": {"linear.weight": _bigtensor(torch)},
        "meta": {"collect_batch": 64, "ppo_epochs": 4},
    }
    torch.save(state, ckpt)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "PASS"
    assert result["details"]["paper_faithful"] is True
    assert result["details"]["collect_batch"] == 64


def test_verify_stage2_pre_patch_meta_returns_warn(tmp_path, monkeypatch):
    """Pre-P0-4 ckpts (collect_batch=4) must surface as WARN, not silent PASS."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": {"linear.weight": _bigtensor(torch)},
        "critic": {"linear.weight": _bigtensor(torch)},
        "meta": {"collect_batch": 4, "ppo_epochs": 4},  # pre-patch
    }
    torch.save(state, ckpt)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "WARN"
    assert result["details"]["paper_faithful"] is False


def test_verify_stage2_missing_meta_returns_pass(tmp_path, monkeypatch):
    """Older ckpts without `meta` are tolerated as PASS (paper_faithful=True)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": {"linear.weight": _bigtensor(torch)},
        "critic": {"linear.weight": _bigtensor(torch)},
    }
    torch.save(state, ckpt)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    # No meta -> tolerated path -> PASS
    assert result["status"] == "PASS"


def test_verify_stage2_missing_policy_returns_fail(tmp_path, monkeypatch):
    """A ckpt with neither policy/actor nor critic/value keys is FAIL."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {"linear.weight": _bigtensor(torch)}  # flat dict, no policy/critic
    torch.save(state, ckpt)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "FAIL"
    assert "policy/critic" in result["details"]["reason"]


def test_verify_stage2_explicit_training_meta_returns_pass(tmp_path, monkeypatch):
    """New-format ckpts written by ``_save_stage2_checkpoint`` carry a
    ``training_meta`` block with ``paper_faithful_p0_4=True``. Phase 1
    must surface that as an *explicit* PASS (not the legacy None-tolerant
    soft pass) so reviewers can see the auditable hyperparams in
    pipeline_status.json."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": {"linear.weight": _bigtensor(torch)},
        "critic": {"linear.weight": _bigtensor(torch)},
        # The new ``meta`` is a *state_dict* (parameter tensors), as
        # written by cts.train.stage2_ppo_train._save_stage2_checkpoint.
        "meta": {"linear.weight": _bigtensor(torch)},
        "training_meta": {
            "step": 10000,
            "total_steps": 10000,
            "collect_batch": 64,
            "ppo_epochs": 4,
            "actor_lr": 3e-5,
            "critic_lr": 1e-4,
            "lambda_halt": 0.05,
            "paper_faithful_p0_4": True,
        },
    }
    torch.save(state, ckpt)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "PASS"
    assert result["details"]["paper_faithful"] is True
    assert result["details"]["explicit_paper_faithful"] is True
    assert result["details"]["has_training_meta"] is True
    assert result["details"]["collect_batch"] == 64
    assert result["details"]["ppo_epochs"] == 4
    assert result["details"]["training_meta"]["lambda_halt"] == 0.05


def test_verify_stage2_training_meta_with_wrong_hparams_warns(tmp_path, monkeypatch):
    """If a future maintainer accidentally trains with collect_batch=32
    (e.g. tweaking config without realising it breaks paper parity),
    Phase 1 must surface this as WARN so the post-stage2 pipeline
    proceeds with a visible flag rather than silently mis-labelling."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    ckpt = tmp_path / "artifacts" / "stage2_meta_value.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": {"linear.weight": _bigtensor(torch)},
        "critic": {"linear.weight": _bigtensor(torch)},
        "training_meta": {
            "collect_batch": 32,  # NOT paper-faithful
            "ppo_epochs": 4,
            "paper_faithful_p0_4": False,
        },
    }
    torch.save(state, ckpt)
    args = SimpleNamespace()
    result = mod.phase_verify_stage2(args)
    assert result["status"] == "WARN"
    assert result["details"]["paper_faithful"] is False
    assert result["details"]["explicit_paper_faithful"] is False
    assert result["details"]["collect_batch"] == 32


# ---------- run_pipeline driver respects --skip-* --------------------------


def test_run_pipeline_skip_all_phases_after_verify(tmp_path, monkeypatch):
    """When all post-verify phases are skipped, the driver writes a status
    JSON with each phase marked SKIP and never invokes the phase fn."""
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    # Force phase 1 to PASS without actually creating a ckpt by
    # monkeypatching phase_verify_stage2 to a stub.
    def _stub_verify(args):
        return {"status": "PASS", "duration_s": 0.0, "details": {}}
    monkeypatch.setattr(mod, "phase_verify_stage2", _stub_verify)

    # Hard guard: the skipped phases must NOT be called.
    def _explode(*a, **kw):
        raise AssertionError("skipped phase invoked")
    monkeypatch.setattr(mod, "phase_table2",      _explode)
    monkeypatch.setattr(mod, "phase_table17",     _explode)
    monkeypatch.setattr(mod, "phase_table19",     _explode)
    monkeypatch.setattr(mod, "phase_zip_rebuild", _explode)

    out = tmp_path / "out"
    args = mod._build_argparser().parse_args([
        "--output-root", str(out),
        "--skip-table2", "--skip-table17", "--skip-table19", "--skip-zip",
    ])
    status = mod.run_pipeline(args)
    assert status["phases"]["verify_stage2"]["status"] == "PASS"
    for phase in ("table2", "table17", "table19", "zip_rebuild"):
        assert status["phases"][phase]["status"] == "SKIP"
    assert (out / "pipeline_status.json").is_file()
    on_disk = json.loads((out / "pipeline_status.json").read_text(encoding="utf-8"))
    assert on_disk["final_verdict"] in ("PASS", "PASS_WITH_WARN", "SKIPPED_ALL")


def test_run_pipeline_phase_1_failure_aborts(tmp_path, monkeypatch):
    """If phase 1 (verify_stage2) FAILs, the driver must not run phases 2-5."""
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    def _stub_fail(args):
        return {"status": "FAIL", "duration_s": 0.0, "details": {"reason": "stub"}}
    monkeypatch.setattr(mod, "phase_verify_stage2", _stub_fail)

    def _explode(*a, **kw):
        raise AssertionError("post-verify phase invoked despite phase 1 FAIL")
    monkeypatch.setattr(mod, "phase_table2",      _explode)
    monkeypatch.setattr(mod, "phase_table17",     _explode)
    monkeypatch.setattr(mod, "phase_table19",     _explode)
    monkeypatch.setattr(mod, "phase_zip_rebuild", _explode)

    out = tmp_path / "out"
    args = mod._build_argparser().parse_args(["--output-root", str(out)])
    status = mod.run_pipeline(args)
    assert status["final_verdict"] == "FAIL_PHASE_1"
    # Subsequent phases must not appear in the dict (driver returns early).
    assert "table2" not in status["phases"]


def test_run_pipeline_phase_exception_does_not_propagate(tmp_path, monkeypatch):
    """A phase that raises must produce a FAIL status JSON, not a crash."""
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    def _stub_pass(args):
        return {"status": "PASS", "duration_s": 0.0, "details": {}}
    monkeypatch.setattr(mod, "phase_verify_stage2", _stub_pass)

    def _raise(*a, **kw):
        raise RuntimeError("phase boom")
    monkeypatch.setattr(mod, "phase_table2", _raise)

    out = tmp_path / "out"
    args = mod._build_argparser().parse_args([
        "--output-root", str(out),
        "--skip-table17", "--skip-table19", "--skip-zip",
    ])
    status = mod.run_pipeline(args)
    assert status["phases"]["verify_stage2"]["status"] == "PASS"
    assert status["phases"]["table2"]["status"] == "FAIL"
    assert "phase boom" in status["phases"]["table2"]["details"]["traceback"]
    # final verdict reflects partial failure
    assert status["final_verdict"] == "PARTIAL_FAIL"
