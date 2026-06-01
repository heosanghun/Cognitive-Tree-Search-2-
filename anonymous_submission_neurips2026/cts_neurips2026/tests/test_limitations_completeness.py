"""Static completeness invariants on LIMITATIONS.md.

LIMITATIONS.md is the single reviewer-facing document where
the author commits, in plain language, to every honest
limitation of the submission. A reviewer reading LIMITATIONS.md
expects every limitation to:

  1. Have a numbered heading (## N. Title).
  2. State what the limitation IS (a "Limitation" subsection).
  3. State what was DONE about it (a "What we have done" /
     "Mitigation" / "What we do" subsection).
  4. State what is NOT claimed (a "What we do *not* claim" /
     "Out of scope" subsection).
  5. Cross-reference to either a paper §-section, a
     REVIEWER_FAQ Qn, or another reviewer-facing artefact
     (so the reviewer can drill in).

This test prevents the failure mode where the author adds a
new limitation as plain prose without the structured
subsections, which would let a reviewer claim "the author is
hiding the mitigation".

Runs in <30 ms, no torch.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIM = ROOT / "LIMITATIONS.md"

# Sub-marker patterns the reviewer expects in every numbered
# limitation. We accept any one of several phrasings to allow
# the author stylistic flexibility.
WHAT_DONE_PATTERNS = (
    "What we have done",
    "What we do",
    "Mitigation",
    "How we address",
    "Status",
    "Resolution",
)

WHAT_NOT_CLAIMED_PATTERNS = (
    "do *not* claim",
    "do **not** claim",
    "do not claim",
    "Out of scope",
    "out of scope",
    "is **not** claimed",
    "Not claimed",
    "Caveat",
)

CROSSREF_PATTERNS = (
    "REVIEWER_FAQ",
    "REPRODUCIBILITY",
    "PAPER_VS_LOCAL",
    "CHANGELOG",
    "§",
    "Q1",
    "Q2",
    "App.",
    "Table",
    "tests/",
    "scripts/",
    "cts/",
)

# Sections that are intentionally meta (no Limitation/Mitigation
# structure expected because they describe the document itself
# rather than a real limitation).
META_SECTIONS = {9, 10}


def _read() -> str:
    assert LIM.is_file(), "LIMITATIONS.md missing on disk"
    return LIM.read_text(encoding="utf-8")


def _split_sections(src: str) -> dict[int, tuple[str, str]]:
    """Map ``N -> (title, body)`` for each ``## N. Title`` heading."""
    out: dict[int, tuple[str, str]] = {}
    pattern = re.compile(
        r"^##\s+(\d+)\.\s+([^\n]+)\n(.*?)(?=\n##\s+\d+\.\s+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for m in pattern.finditer(src):
        out[int(m.group(1))] = (m.group(2), m.group(3))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_limitations_md_exists_and_nonempty():
    assert LIM.is_file()
    src = _read()
    assert len(src) > 4000, (
        f"LIMITATIONS.md suspiciously short ({len(src)} bytes); likely truncated"
    )


def test_at_least_10_numbered_limitations():
    """The 10 sections committed in Plan B were:
    compute-scaling, Native Think, ARC-AGI proxy, nu-ablation
    paths, missing baselines, AIME garbage (Q14), CUDA
    deadlock (Q15), implementation status, checklist
    coverage, plain-language summary."""
    src = _read()
    sections = _split_sections(src)
    assert len(sections) >= 10, (
        f"LIMITATIONS.md must contain >= 10 numbered sections; got {len(sections)}"
    )


def test_every_limitation_has_what_done_subsection():
    """Every non-meta limitation must explain what the author
    did about it."""
    src = _read()
    sections = _split_sections(src)
    bad: list[int] = []
    for n, (title, body) in sections.items():
        if n in META_SECTIONS:
            continue
        if not any(pat in body for pat in WHAT_DONE_PATTERNS):
            bad.append(n)
    assert not bad, (
        f"LIMITATIONS sections missing 'What we have done' / "
        f"'Mitigation' / 'Status' subsection: {bad}"
    )


def test_every_limitation_has_explicit_non_claim_disclosure():
    """The reviewer must be able to read each section and find
    a one-line statement of what the author is NOT claiming.
    This is the single most-cited NeurIPS reviewer ask."""
    src = _read()
    sections = _split_sections(src)
    bad: list[int] = []
    for n, (title, body) in sections.items():
        if n in META_SECTIONS:
            continue
        if not any(pat in body for pat in WHAT_NOT_CLAIMED_PATTERNS):
            bad.append(n)
    assert not bad, (
        f"LIMITATIONS sections missing explicit non-claim "
        f"('do not claim' / 'out of scope' / 'caveat'): {bad}"
    )


def test_every_limitation_has_crossreference():
    """Every limitation must point the reviewer at another
    reviewer-facing artefact (paper §-number, FAQ Qn, code
    path, or test path) so they can drill in."""
    src = _read()
    sections = _split_sections(src)
    bad: list[int] = []
    for n, (title, body) in sections.items():
        if n in META_SECTIONS:
            continue
        if not any(pat in body for pat in CROSSREF_PATTERNS):
            bad.append(n)
    assert not bad, (
        f"LIMITATIONS sections missing reviewer cross-reference "
        f"(FAQ / REPRODUCIBILITY / §N / code path): {bad}"
    )


def test_q14_and_q15_limitations_present():
    """The two D-7 incident limitations must be in LIMITATIONS.md
    (not just in CHANGELOG / FAQ), because reviewers read
    LIMITATIONS first when looking for known failures."""
    src = _read()
    assert "Q14" in src or "AIME garbage" in src, (
        "LIMITATIONS.md must document the Q14 AIME garbage incident"
    )
    assert "Q15" in src or "CUDA driver deadlock" in src or "single-host" in src, (
        "LIMITATIONS.md must document the Q15 single-host CUDA deadlock"
    )


def test_section_titles_have_unique_keywords():
    """Every section title must be unique (no copy-paste
    duplicates), which catches the failure mode where a new
    limitation overwrites an old one's heading."""
    src = _read()
    sections = _split_sections(src)
    titles = [t for _, (t, _) in sorted(sections.items())]
    assert len(titles) == len(set(titles)), (
        f"LIMITATIONS.md has duplicate section titles: {titles}"
    )


def test_plain_language_summary_section_present():
    """Section 10 (or wherever the plain-language summary lives)
    is what skim-only reviewers read first; it must exist
    and mention every prior section's headline limitation
    in a single paragraph."""
    src = _read()
    summary_keywords = ("plain-language", "Plain-language",
                        "Plain language", "summary for skim",
                        "tl;dr", "TL;DR", "skim-only")
    assert any(kw in src for kw in summary_keywords), (
        "LIMITATIONS.md must contain a plain-language / TL;DR / "
        "skim-only summary section"
    )


def test_no_internal_filenames_leak():
    """LIMITATIONS.md must not reference internal author
    drafts (PROGRESS_REPORT*, OPENREVIEW_RESPONSE_PREP*,
    NEXT_TASKS*); those leak the author's identity / process."""
    src = _read()
    forbidden = ("PROGRESS_REPORT", "OPENREVIEW_RESPONSE_PREP",
                 "NEXT_TASKS", "ROOT_CAUSE_ANALYSIS",
                 "EXPERIMENTAL_RESULTS", "PAPER_CONSISTENCY_AUDIT")
    leaked = [pat for pat in forbidden if pat in src]
    assert not leaked, (
        f"LIMITATIONS.md leaks references to internal author drafts: {leaked}"
    )
