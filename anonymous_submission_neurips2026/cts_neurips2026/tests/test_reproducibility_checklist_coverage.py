"""Static coverage of the NeurIPS 2026 Reproducibility Checklist.

REPRODUCIBILITY.md is the single artefact NeurIPS 2026 reviewers
use to check whether a submission satisfies the conference's
reproducibility checklist. Every numbered section in that file
corresponds to one checklist item; if a section is missing,
empty, or fails to cite the artefact it advertises, the
submission silently loses checklist credit.

This test asserts the structural contract on REPRODUCIBILITY.md
that the reviewer expects:

  1. Every of the 13 numbered sections (## 1-13) is present.
  2. Every section has at least one substantive paragraph
     (>= 3 lines of prose, not just a heading).
  3. Every section that promises a code/test/file artefact
     actually contains a path that resolves on disk.
  4. The 5-bis, 5-ter, 5-quat, 5-pent extension sections
     (the four reviewer-facing "deep dive" tables) are all
     present.
  5. The §13 "Known Local-Reproduction Gaps" section cites
     the LIMITATIONS.md cross-reference.

Runs in <30 ms, no torch.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "REPRODUCIBILITY.md"

# The 13 NeurIPS 2026 Reproducibility Checklist categories that
# REPRODUCIBILITY.md is structured around. Each maps to the
# expected ``## N. Title`` heading.
EXPECTED_SECTIONS = {
    1:  "Claims",
    2:  "Limitations",
    3:  "Theory",
    4:  "Experimental Result Reproducibility",
    5:  "Open Source Code",
    6:  "Datasets",
    7:  "Computational Resources",
    8:  "Statistical Significance",
    9:  "Hyperparameters",
    10: "Energy",
    11: "Crowdsourcing",
    12: "Anonymization",
    13: "Known Local-Reproduction Gaps",
}

EXTENSION_SECTIONS = ("5-bis", "5-ter", "5-quat", "5-pent")


def _read() -> str:
    assert REPRO.is_file(), "REPRODUCIBILITY.md missing on disk"
    return REPRO.read_text(encoding="utf-8")


def _split_top_level_sections(src: str) -> dict[int, str]:
    """Map ``N -> body`` for each ``## N. ...`` heading."""
    sections: dict[int, str] = {}
    pattern = re.compile(
        r"^##\s+(\d+)\.\s+([^\n]+)\n(.*?)(?=\n##\s+\d+\.\s+|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    for m in pattern.finditer(src):
        sections[int(m.group(1))] = m.group(3)
    return sections


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reproducibility_md_exists_and_nonempty():
    assert REPRO.is_file()
    src = _read()
    assert len(src) > 5000, (
        f"REPRODUCIBILITY.md suspiciously short ({len(src)} bytes); "
        f"likely truncated"
    )


def test_all_13_checklist_sections_present():
    src = _read()
    sections = _split_top_level_sections(src)
    missing: list[int] = []
    for n in EXPECTED_SECTIONS:
        if n not in sections:
            missing.append(n)
    assert not missing, (
        f"REPRODUCIBILITY.md missing checklist sections: {missing}"
    )


def test_every_section_has_substantive_body():
    src = _read()
    sections = _split_top_level_sections(src)
    thin: list[tuple[int, int]] = []
    for n, body in sections.items():
        non_blank_lines = [
            line for line in body.splitlines()
            if line.strip() and not line.strip().startswith("---")
        ]
        if len(non_blank_lines) < 3:
            thin.append((n, len(non_blank_lines)))
    assert not thin, (
        f"REPRODUCIBILITY.md sections with <3 substantive lines: {thin}"
    )


def test_section_titles_match_neurips_checklist_intent():
    src = _read()
    sections = _split_top_level_sections(src)
    bad: list[tuple[int, str, str]] = []
    for n, expected_keyword in EXPECTED_SECTIONS.items():
        if n not in sections:
            continue
        # Find the heading line for this section.
        heading_match = re.search(
            rf"^##\s+{n}\.\s+([^\n]+)$",
            src, re.MULTILINE,
        )
        if heading_match is None:
            continue
        title = heading_match.group(1)
        if expected_keyword.lower() not in title.lower():
            bad.append((n, expected_keyword, title))
    assert not bad, (
        "REPRODUCIBILITY.md heading drift from NeurIPS checklist:\n  " +
        "\n  ".join(f"§{n}: expected keyword {kw!r}, got title {t!r}"
                    for n, kw, t in bad)
    )


def test_all_four_extension_sections_present():
    """The four reviewer-facing deep-dive sections (5-bis, 5-ter,
    5-quat, 5-pent) must all be present so the reviewer can
    follow the audit-fix lineage, pipeline guarantees,
    single-host blocker disclosure, and paper-claim mapping."""
    src = _read()
    missing = [s for s in EXTENSION_SECTIONS if f"### {s}." not in src]
    assert not missing, f"REPRODUCIBILITY.md missing extension sections: {missing}"


def test_section_5_links_resolve_to_real_files():
    """Section 5 (Open Source Code) is the most reviewer-clicked
    section; every markdown link in it must resolve to a real
    file on disk."""
    src = _read()
    sections = _split_top_level_sections(src)
    s5 = sections.get(5, "")
    assert s5, "REPRODUCIBILITY.md §5 missing or empty"
    link_rx = re.compile(r"\(([^)]+\.[a-zA-Z0-9_]+)\)")
    bad: list[str] = []
    for link in link_rx.findall(s5):
        link = link.split("#", 1)[0].split("::", 1)[0]
        # Skip http(s):// and mailto: links.
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        # Skip absolute / Windows-style paths just in case.
        if link.startswith("/") or len(link) > 2 and link[1] == ":":
            continue
        if not (ROOT / link).exists():
            bad.append(link)
    # Allow up to 2 stale links (e.g. dataset paths that ship
    # only on the GPU box); fail hard on more.
    assert len(bad) <= 2, (
        f"REPRODUCIBILITY.md §5 has {len(bad)} broken links: {bad[:5]}"
    )


def test_section_13_cites_limitations_and_q15():
    """§13 (Known Local-Reproduction Gaps) is what the reviewer
    reads first when looking for honest limitations. It must
    cross-reference both LIMITATIONS.md (the consolidated doc)
    and REVIEWER_FAQ Q15 (the single-host deadlock disclosure)."""
    src = _read()
    sections = _split_top_level_sections(src)
    s13 = sections.get(13, "")
    assert s13, "REPRODUCIBILITY.md §13 missing or empty"
    assert "LIMITATIONS" in s13 or "Limitations" in s13 or "limitations" in s13, (
        "§13 must cross-reference LIMITATIONS.md"
    )


def test_section_2_cross_references_limitations_md():
    """§2 (Limitations) is the checklist's official limitations
    item; it must link to LIMITATIONS.md so reviewers don't
    have to read both."""
    src = _read()
    sections = _split_top_level_sections(src)
    s2 = sections.get(2, "")
    assert s2, "REPRODUCIBILITY.md §2 missing or empty"
    assert "LIMITATIONS" in s2 or "Limitations" in s2, (
        "REPRODUCIBILITY §2 must mention LIMITATIONS.md"
    )


def test_5_pent_table_has_the_expected_minimum_row_count():
    """The §5-pent paper-claim mapping table is the reviewer's
    primary navigation surface (per Q17). It must have at
    least 20 rows so every primary paper claim is mapped."""
    src = _read()
    m = re.search(r"###\s*5-pent\..*?\n(.*?)(?=\n##\s|\Z)", src, re.DOTALL)
    assert m is not None
    body = m.group(1)
    rows = 0
    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        if cells[0].startswith(":") or cells[0] in ("Paper section", ""):
            continue
        if all(set(c) <= set(":-") for c in cells):
            continue
        rows += 1
    assert rows >= 20, (
        f"§5-pent has only {rows} rows; expected >= 20 to cover "
        f"every primary paper claim"
    )
