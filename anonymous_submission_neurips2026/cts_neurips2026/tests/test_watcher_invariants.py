"""Text-level invariants for ``scripts/wait_and_run_pipeline.ps1``.

PowerShell scripts are awkward to invoke from CI (no Mac/Linux
runner has PowerShell on the default path), but we can still defend
the *contract* of the watcher with structural assertions:

* The watcher must declare a max-wait timeout and write a marker
  file when Stage 2 hangs.
* The watcher must detect a Stage 2 *crash* (process gone but ckpt
  stale) and write a STAGE2_CRASHED marker rather than looping
  forever.
* The watcher must update a heartbeat file each poll so an external
  operator can see at a glance whether automation is alive.
* The watcher must exit non-zero when the pipeline is not launched.

These guarantees protect the D11->D12 hand-off: a silent watcher
loop is the single most dangerous failure mode (Stage 2 crashes,
nobody notices, deadline missed).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "wait_and_run_pipeline.ps1"


def _read():
    assert SCRIPT.is_file(), f"{SCRIPT} missing"
    return SCRIPT.read_text(encoding="utf-8")


def test_watcher_script_exists_and_is_nontrivial():
    txt = _read()
    assert len(txt) > 1000, "watcher script suspiciously small"
    assert txt.lower().startswith("# stage 2 watcher"), (
        "watcher script must lead with its purpose comment"
    )


def test_watcher_declares_max_wait_timeout():
    txt = _read()
    assert "$MaxWaitMin" in txt, (
        "watcher must declare a $MaxWaitMin parameter to bound runtime"
    )
    assert "STAGE2_TIMEOUT" in txt, (
        "watcher must write a STAGE2_TIMEOUT marker on hang"
    )


def test_watcher_detects_crash():
    """Process gone + ckpt stale = crash; do not loop forever."""
    txt = _read()
    assert "STAGE2_CRASHED" in txt, (
        "watcher must surface Stage 2 crashes via a marker file"
    )
    # The crash branch must check that ckpt is *not* ready.
    assert "(-not $alive) -and (-not $ckptReady)" in txt, (
        "watcher missing the (gone + stale) crash detection branch"
    )


def test_watcher_writes_heartbeat():
    txt = _read()
    assert "$HeartbeatPath" in txt, (
        "watcher must declare a heartbeat path parameter"
    )
    assert "Write-Heartbeat" in txt, (
        "watcher must call its heartbeat writer each poll"
    )
    # The default heartbeat path lives under results/ so external
    # tools can find it without knowing repo layout.
    assert "results/.watcher_heartbeat.json" in txt or \
           "results\\.watcher_heartbeat.json" in txt


def test_watcher_exits_nonzero_on_non_ok():
    txt = _read()
    assert 'exit 2' in txt, (
        "watcher must exit non-zero when not launching pipeline"
    )
    assert "OK_LAUNCH_PIPELINE" in txt, (
        "watcher must distinguish the success branch by reason"
    )


def test_watcher_forwards_pipeline_args():
    """The watcher must forward --seeds/--device/--output-root to
    run_post_stage2_pipeline.py exactly; otherwise reviewers see
    seed=1 instead of the paper's 5 seeds."""
    txt = _read()
    assert "--seeds" in txt and "$Seeds" in txt
    assert "--device" in txt and "$Device" in txt
    assert "--output-root" in txt and "$OutputRoot" in txt
    assert "scripts/run_post_stage2_pipeline.py" in txt


def test_watcher_forwards_d7_limit_knobs():
    """D-7 partial-save patch: watcher must forward --table2-limit /
    --table17-limit / --skip-verify so the post-reboot auto-launch
    matches the reviewer-facing canonical command in
    results/table2/PAPER_VS_LOCAL.md without manual editing."""
    txt = _read()
    assert "$Table2Limit" in txt, (
        "watcher must declare $Table2Limit param for D-7 partial-save runs"
    )
    assert "$Table17Limit" in txt, (
        "watcher must declare $Table17Limit param for D-7 partial-save runs"
    )
    assert "--table2-limit" in txt, (
        "watcher must forward --table2-limit to the pipeline"
    )
    assert "--table17-limit" in txt, (
        "watcher must forward --table17-limit to the pipeline"
    )
    assert "$SkipVerify" in txt and "--skip-verify" in txt, (
        "watcher must forward --skip-verify when ckpt pre-validated"
    )
    # Both limits must be guarded by `-gt 0` so default 0 means
    # "full benchmark" (paper-faithful), preserving backward compat.
    assert "$Table2Limit -gt 0" in txt
    assert "$Table17Limit -gt 0" in txt
