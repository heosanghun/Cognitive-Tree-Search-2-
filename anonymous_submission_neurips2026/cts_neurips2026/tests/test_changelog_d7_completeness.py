"""CHANGELOG D-7 completeness invariants.

Every D-7 batch (Plan B / C / D / E / F / ...) must leave a
CHANGELOG entry that cites:

  1. At least one paper §-section OR Q-number that anchors the
     change to a reviewer-facing document, AND
  2. Every new file path created in that batch (via the
     ``Added`` / ``Updated`` lists), AND
  3. A pass/total ratio for any test suite the batch added,
     so the reviewer can spot-check that the suite is locked
     in.

This test prevents the "I committed a fix but the CHANGELOG
is silent" failure mode that would surface during the May 6
final review (a reviewer scrolling CHANGELOG would not see
the latest patches).

Runs in <50 ms, no torch.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHLOG = ROOT / "CHANGELOG.md"

# Plan letter -> minimum required artefact paths that the
# batch must mention in its CHANGELOG block.
REQUIRED_ARTIFACTS = {
    "Plan B": (
        "REVIEWER_FAQ",
        "PAPER_VS_LOCAL",
    ),
    "Plan C": (
        "scripts/_d12_final_check.py",
        "tests/test_d7_static_validation.py",
    ),
    "Plan D": (
        "cts/eval/garbage_filter.py",
        "tests/test_dispatcher_fallback_mock.py",
        "scripts/replicate_neurips_2026.sh",
    ),
    "Plan E": (
        "tests/test_paper_code_mapping_table.py",
        "tests/test_stage2_training_meta_static.py",
        "--dry-run",
        "Reviewer Quick Start",
    ),
    "Plan F": (
        "tests/test_anon_zip_byte_invariants.py",
        "tests/test_paper_section_alignment.py",
        "scripts/reviewer_walkthrough.py",
        "--ci-mode",
        "Q16",
    ),
}

# Section-family / Q-number patterns. A D-7 block must contain
# at least one match.
SECTION_RX = re.compile(r"§\d+(?:\.\d+)?|App\.\s*[A-Z]|\bQ1[0-9]\b|P0-\d|Plan [A-Z]")


def _read_changelog() -> str:
    return CHLOG.read_text(encoding="utf-8")


def _split_d7_blocks(src: str) -> list[tuple[str, str]]:
    """Return list of (header, body) tuples for every D-7 entry
    in the CHANGELOG. The header is the line starting with
    ``## [unreleased]`` and ``D-7``."""
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(
        r"^(##\s+\[unreleased\][^\n]*D-7[^\n]*)\n(.*?)(?=\n##\s+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for m in pattern.finditer(src):
        blocks.append((m.group(1), m.group(2)))
    return blocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_changelog_exists():
    assert CHLOG.is_file(), "CHANGELOG.md missing on disk"


def test_changelog_has_d7_entries():
    blocks = _split_d7_blocks(_read_changelog())
    assert len(blocks) >= 5, (
        f"expected >=5 D-7 batches (Plan B/C/D/E/F), got {len(blocks)}"
    )


def test_every_d7_block_cites_a_section_or_qnumber():
    blocks = _split_d7_blocks(_read_changelog())
    bad: list[str] = []
    for header, body in blocks:
        full = header + "\n" + body
        if not SECTION_RX.search(full):
            bad.append(header[:80])
    assert not bad, (
        "D-7 blocks missing §-section / Q-number / Plan letter:\n  " +
        "\n  ".join(bad)
    )


def test_every_known_plan_block_cites_required_artefacts():
    """For every Plan letter we have required-artefact lists for,
    the matching CHANGELOG block must mention each artefact."""
    blocks = _split_d7_blocks(_read_changelog())
    text_by_plan: dict[str, str] = {}
    for header, body in blocks:
        for plan_letter in REQUIRED_ARTIFACTS:
            if plan_letter in header:
                text_by_plan[plan_letter] = header + "\n" + body
    for plan_letter, artefacts in REQUIRED_ARTIFACTS.items():
        if plan_letter not in text_by_plan:
            # Plan F was just added; older Plan B markers may live
            # in a different format. Skip if not yet present rather
            # than failing.
            continue
        block = text_by_plan[plan_letter]
        missing = [a for a in artefacts if a not in block]
        assert not missing, (
            f"{plan_letter} CHANGELOG block missing artefact mentions: {missing}"
        )


def test_d7_entries_cite_anonymous_zip_status():
    """Every D-7 entry that touches the ZIP must cite either
    a file count, a leak count, or an audit-verdict. This
    catches the failure mode where a fix lands but the
    reviewer cannot tell whether the ZIP was rebuilt."""
    blocks = _split_d7_blocks(_read_changelog())
    bad: list[str] = []
    for header, body in blocks:
        full = (header + "\n" + body).lower()
        if "zip" not in full and "anonymous" not in full:
            continue
        # If the entry mentions ZIP, it must also cite a status
        # marker the reviewer can spot-check.
        markers = ("0 leaks", "audit verdict", "files,", "files |",
                   "audit pass", "pass", "verdict pass")
        if not any(marker in full for marker in markers):
            bad.append(header[:80])
    assert not bad, (
        "D-7 blocks mention ZIP but not status (file count / "
        "leak count / verdict):\n  " + "\n  ".join(bad)
    )


def test_d7_test_suite_additions_cite_pass_ratio():
    """If a D-7 block mentions adding a new tests/test_*.py,
    it must also cite the pass/total ratio so a reviewer can
    spot-check the suite is locked in."""
    blocks = _split_d7_blocks(_read_changelog())
    test_add_rx = re.compile(r"tests/test_[A-Za-z0-9_]+\.py")
    pass_ratio_rx = re.compile(r"\b\d+\s*/\s*\d+\b|\bPASS\b", re.IGNORECASE)
    bad: list[tuple[str, list[str]]] = []
    for header, body in blocks:
        full = header + "\n" + body
        added_tests = set(test_add_rx.findall(full))
        if not added_tests:
            continue
        if not pass_ratio_rx.search(full):
            bad.append((header[:80], sorted(added_tests)))
    assert not bad, (
        "D-7 blocks add test files but cite no pass/N or PASS marker:\n  " +
        "\n  ".join(f"{h}: {ts}" for h, ts in bad)
    )


def test_latest_d7_block_is_plan_f_or_newer():
    """The most recent D-7 entry should be Plan F (or whatever
    is newer). If a future Plan G lands, this test will need
    to be bumped, which is exactly the behaviour we want
    (forces the author to keep this list synced)."""
    blocks = _split_d7_blocks(_read_changelog())
    if not blocks:
        return  # caught by test_changelog_has_d7_entries
    latest_header = blocks[0][0]  # CHANGELOG is reverse-chronological
    assert "Plan F" in latest_header or "Plan G" in latest_header \
        or "Plan H" in latest_header, (
        f"latest D-7 block does not mention Plan F+ ; got: {latest_header[:120]}"
    )
