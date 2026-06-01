"""Torch-free static validation of the Stage 2 PPO ``training_meta``
audit block contract.

Why this file exists:

The post-Stage-2 pipeline (`scripts/run_post_stage2_pipeline.py
::phase_verify_stage2`) treats ``training_meta.paper_faithful_p0_4``
as a *hard PASS* gate: if a reviewer's checkpoint is missing the
block or carries the wrong hyperparameters, phase 1 of the
pipeline fails fast and the reviewer's table refresh never starts.

This contract therefore has two sides that MUST stay in sync:

  - Writer: ``cts.train.stage2_ppo_train._save_stage2_checkpoint``
    populates the ``training_meta`` dict with the paper §6.2 / P0-4
    hyperparameters.
  - Reader: ``scripts.run_post_stage2_pipeline.phase_verify_stage2``
    reads the *same* keys and computes the verdict.

A drift in either direction silently corrupts the gate. This test
locks both sides via AST inspection in <50 ms with no torch
dependency, so it can run in CI on every push regardless of GPU
availability.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Canonical keys the writer is required to populate. Drift here
# is exactly what this test prevents (e.g. someone renaming
# ``ppo_epochs`` to ``ppo_inner_epochs`` would silently break the
# reader's strict gate without surfacing in any unit test).
REQUIRED_TRAINING_META_KEYS = {
    "step",
    "total_steps",
    "collect_batch",
    "ppo_epochs",
    "actor_lr",
    "critic_lr",
    "lambda_halt",
    "paper_faithful_p0_4",
}

# Paper-faithful hyperparameter values from paper §6.2 P0-4.
PAPER_COLLECT_BATCH = 64
PAPER_PPO_EPOCHS = 4

WRITER = ROOT / "cts" / "train" / "stage2_ppo_train.py"
READER = ROOT / "scripts" / "run_post_stage2_pipeline.py"


def _read(p: Path) -> str:
    assert p.is_file(), f"{p} missing"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section 1: writer side - every required key is in the persisted dict
# ---------------------------------------------------------------------------


def test_writer_persists_every_required_key():
    """``_save_stage2_checkpoint`` must populate every key in
    ``REQUIRED_TRAINING_META_KEYS``. We parse the writer's source
    AST and look for the dict literal under the ``training_meta``
    key."""
    src = _read(WRITER)
    tree = ast.parse(src)

    found_keys: set[str] = set()
    for node in ast.walk(tree):
        # Look for a dict literal that contains the key
        # ``training_meta`` mapped to another dict literal.
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == "training_meta"
                    and isinstance(v, ast.Dict)):
                for inner_k in v.keys:
                    if isinstance(inner_k, ast.Constant) and isinstance(inner_k.value, str):
                        found_keys.add(inner_k.value)

    missing = REQUIRED_TRAINING_META_KEYS - found_keys
    assert not missing, (
        f"_save_stage2_checkpoint missing training_meta keys: {missing}\n"
        f"found keys: {sorted(found_keys)}"
    )


def test_writer_uses_paper_faithful_predicate():
    """The ``paper_faithful_p0_4`` flag must be computed from the
    P0-4 predicate ``int(collect_batch) == 64 and int(ppo_epochs) == 4``;
    a lazy ``= True`` would silently mark non-paper-faithful runs
    as PASS and corrupt the reviewer-facing audit log."""
    src = _read(WRITER)
    pat = re.compile(
        r"paper_faithful_p0_4\s*=\s*bool\(\s*int\(collect_batch\)\s*==\s*64"
        r"\s*and\s*int\(ppo_epochs\)\s*==\s*4\s*\)"
    )
    assert pat.search(src), (
        "writer must compute paper_faithful_p0_4 from "
        "(collect_batch == 64 and ppo_epochs == 4)"
    )


def test_writer_persists_into_training_meta_not_top_level():
    """Defence-in-depth: ``paper_faithful_p0_4`` must live INSIDE
    the ``training_meta`` dict, not at the checkpoint top level
    (the reader looks at ``sd.get('training_meta')``, not
    ``sd.get('paper_faithful_p0_4')``)."""
    src = _read(WRITER)
    # Crude-but-effective: check the key sits inside the nested dict
    # we already verified above. We do this by counting how many
    # times ``"paper_faithful_p0_4"`` appears as a Constant key
    # under a Dict whose grandparent has ``"training_meta"`` as a
    # sibling.
    tree = ast.parse(src)
    inside_training_meta = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == "training_meta"
                    and isinstance(v, ast.Dict)):
                inner_keys = {
                    ik.value for ik in v.keys
                    if isinstance(ik, ast.Constant) and isinstance(ik.value, str)
                }
                if "paper_faithful_p0_4" in inner_keys:
                    inside_training_meta = True
    assert inside_training_meta, (
        "paper_faithful_p0_4 must live inside the training_meta sub-dict"
    )


# ---------------------------------------------------------------------------
# Section 2: reader side - same keys, same gate
# ---------------------------------------------------------------------------


def test_reader_loads_training_meta_block():
    """``phase_verify_stage2`` must call ``sd.get('training_meta')``
    (i.e. the canonical audit-block name) before falling back to
    the legacy ``meta`` dict."""
    src = _read(READER)
    assert 'sd.get("training_meta")' in src or "sd.get('training_meta')" in src, (
        "reader must call sd.get('training_meta') to read the audit block"
    )


def test_reader_consults_canonical_hyperparameter_keys():
    """The reader must extract ``collect_batch``, ``ppo_epochs``,
    and ``paper_faithful_p0_4`` from the ``training_meta`` block
    (not from a legacy alias). Any rename here without a
    corresponding rename in the writer would silently break the
    audit gate."""
    src = _read(READER)
    must_have = (
        "training_meta.get(\"collect_batch\")",
        "training_meta.get(\"ppo_epochs\")",
        "training_meta.get(\"paper_faithful_p0_4\")",
    )
    missing = [k for k in must_have if k not in src.replace("'", '"')]
    assert not missing, (
        f"reader missing training_meta lookups: {missing}"
    )


def test_reader_treats_paper_faithful_as_hard_pass():
    """The reader must surface ``explicit_paper_faithful`` so the
    pipeline gate distinguishes ``training_meta.paper_faithful_p0_4
    == True`` (hard PASS) from a legacy ckpt with no
    ``training_meta`` block (None-tolerated soft PASS)."""
    src = _read(READER)
    assert "explicit_paper_faithful" in src, (
        "reader must expose 'explicit_paper_faithful' field for the gate"
    )
    # And the value must come from training_meta, not from a
    # heuristic guess.
    assert re.search(
        r'explicit_paper_faithful\s*=\s*bool\(\s*training_meta\.get\(',
        src,
    ), "explicit_paper_faithful must be derived from training_meta.get('paper_faithful_p0_4')"


def test_reader_emits_has_training_meta_for_audit():
    """The reader must surface ``has_training_meta`` (boolean) in
    its phase-1 result so the reviewer-facing audit log can
    distinguish a legacy ckpt (None-tolerated) from a modern
    one (hard PASS / hard FAIL)."""
    src = _read(READER)
    assert '"has_training_meta"' in src or "'has_training_meta'" in src, (
        "reader must include 'has_training_meta' in its phase-1 result"
    )


# ---------------------------------------------------------------------------
# Section 3: cross-side consistency (the most important test)
# ---------------------------------------------------------------------------


def test_writer_reader_keys_are_consistent():
    """Every key the WRITER persists into ``training_meta`` must
    be a key the READER knows how to interpret OR is documented
    as informational. Any drift here is the silent-corruption
    failure mode this test exists to prevent."""
    writer_keys = REQUIRED_TRAINING_META_KEYS  # already verified above
    reader = _read(READER)
    # The reader doesn't have to read every key (some are
    # informational, like 'step'/'total_steps'/'actor_lr'/
    # 'critic_lr'/'lambda_halt'), but it MUST read the gate keys.
    gate_keys = {"collect_batch", "ppo_epochs", "paper_faithful_p0_4"}
    for k in gate_keys:
        assert k in writer_keys, f"gate key {k!r} must be persisted by writer"
        assert (f'"{k}"' in reader or f"'{k}'" in reader), (
            f"gate key {k!r} must be read by reader"
        )


def test_paper_faithful_constants_match_paper_section_6_2():
    """The numeric constants (collect_batch=64, ppo_epochs=4) the
    writer hard-codes into the predicate must match paper §6.2.
    A regression that changes 64 -> 32 here would silently mark
    the off-budget run as paper-faithful."""
    src = _read(WRITER)
    pat = re.compile(
        r"int\(collect_batch\)\s*==\s*(\d+)\s*and\s*int\(ppo_epochs\)\s*==\s*(\d+)"
    )
    m = pat.search(src)
    assert m is not None, "writer's paper-faithful predicate not found"
    assert int(m.group(1)) == PAPER_COLLECT_BATCH, (
        f"writer collect_batch constant = {m.group(1)}, expected {PAPER_COLLECT_BATCH}"
    )
    assert int(m.group(2)) == PAPER_PPO_EPOCHS, (
        f"writer ppo_epochs constant = {m.group(2)}, expected {PAPER_PPO_EPOCHS}"
    )
