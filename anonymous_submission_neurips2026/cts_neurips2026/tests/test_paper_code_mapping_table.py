"""Static validation of REPRODUCIBILITY.md §5-pent paper-claim
&rarr; code-line &rarr; regression-test mapping table.

This test exists to make the §5-pent table *executable*:
it parses the markdown table, extracts every (impl, test) cell,
and asserts that every referenced file exists on disk and that
every referenced symbol exists in the implementation file.

Why it matters for NeurIPS:

If a reviewer follows §5-pent to verify a paper claim and the
linked file or symbol does not exist, that is the worst possible
single-signal reproducibility regression. This test catches such
drift in <50 ms with no torch dependency, so it can run in CI on
every push regardless of GPU availability.

Coverage:

- Every Markdown link target in column 3 (Implementation file)
  must resolve to a file under the repo root.
- Every backtick `symbol` after a `(`...`)` in column 3 must
  exist as an AST FunctionDef / ClassDef name in that file
  (best-effort; symbols inside multi-file paths are checked
  against the first listed file only).
- Every backtick test path in column 4 (Regression test) must
  resolve to a file under ``tests/`` AND must contain at least
  one ``def test_`` (best-effort static parse).

This test is intentionally **strict-but-not-pedantic**: a row
that lists 3 candidate test files only needs *one* of them to
exist for the row to pass, but every listed implementation
path must exist because reviewers click those directly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "REPRODUCIBILITY.md"


# ---------------------------------------------------------------------------
# Markdown table parser (intentionally tiny, no markdown dep)
# ---------------------------------------------------------------------------


def _extract_5pent_rows() -> list[dict[str, str]]:
    """Return a list of {'section', 'claim', 'impl', 'tests'} rows
    parsed out of the §5-pent table in REPRODUCIBILITY.md."""
    src = REPRO.read_text(encoding="utf-8")
    # Find the §5-pent section.
    m = re.search(r"###\s*5-pent\..*?\n(.*?)(?=\n##\s|\Z)", src, re.DOTALL)
    assert m is not None, "REPRODUCIBILITY.md missing §5-pent section"
    body = m.group(1)

    rows: list[dict[str, str]] = []
    for line in body.splitlines():
        # Match data rows that start with '|' and have at least 4 cells.
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        # Skip header and separator rows.
        if cells[0].startswith(":") or cells[0] in ("Paper section", ""):
            continue
        if all(set(c) <= set(":-") for c in cells):
            continue
        rows.append({
            "section": cells[0],
            "claim": cells[1],
            "impl": cells[2],
            "tests": cells[3],
        })
    return rows


def _extract_md_link_targets(md_cell: str) -> list[str]:
    """Pull every ``[label](path)`` target out of a markdown cell
    and return the path components."""
    return re.findall(r"\(([^)]+\.[a-zA-Z0-9]+)\)", md_cell)


def _extract_backtick_targets(md_cell: str) -> list[str]:
    """Pull every ``` `tests/test_*.py` ``` (or any path-like
    backticked token ending with `.py`, optionally followed by
    a ``::test_function`` qualifier) out of a markdown cell."""
    return re.findall(r"`([^`]+?\.py(?:::[A-Za-z_][A-Za-z0-9_]*)?)`", md_cell)


def _extract_backtick_symbols(md_cell: str) -> list[str]:
    """Pull symbol-like tokens from `(`...`)` clauses, e.g.
    ``(`MetaPolicy`)`` or ``(`puct_score`, `select_action`)``."""
    out: list[str] = []
    for clause in re.findall(r"\(([^)]+)\)", md_cell):
        for sym in re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", clause):
            out.append(sym)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_5pent_table_is_non_empty():
    """The §5-pent table must have at least 20 mapped paper claims
    (a regression that drops the table to a single row would be
    silently catastrophic for reviewer trust)."""
    rows = _extract_5pent_rows()
    assert len(rows) >= 20, (
        f"§5-pent table shrank to {len(rows)} rows; expected >= 20"
    )


def test_5pent_every_impl_file_exists():
    """Every markdown link in column 3 (Implementation file) must
    resolve to a file under the repo root."""
    rows = _extract_5pent_rows()
    missing: list[tuple[str, str]] = []
    for row in rows:
        for path in _extract_md_link_targets(row["impl"]):
            if (ROOT / path).is_file() or (ROOT / path).is_dir():
                continue
            missing.append((row["section"], path))
    assert not missing, (
        f"§5-pent links to non-existent implementation files: {missing}"
    )


def test_5pent_every_test_path_resolves_to_at_least_one_real_file():
    """Each row may list 1+ candidate test files (comma-separated
    in column 4). At least ONE of those files must exist on disk
    (otherwise the row's reproducibility evidence is broken)."""
    rows = _extract_5pent_rows()
    bad: list[str] = []
    for row in rows:
        candidates = _extract_backtick_targets(row["tests"])
        if not candidates:
            bad.append(f"{row['section']}: no backtick test paths in column 4")
            continue
        # Each candidate may contain a ::test_function suffix;
        # split that off to get just the file path.
        any_exists = False
        for c in candidates:
            file_part = c.split("::", 1)[0]
            if (ROOT / file_part).is_file():
                any_exists = True
                break
        if not any_exists:
            bad.append(f"{row['section']}: no candidate test exists -> {candidates}")
    assert not bad, "§5-pent test references unresolved:\n  " + "\n  ".join(bad)


def test_5pent_test_files_actually_define_tests():
    """Spot-check: every test file referenced from the §5-pent
    table that exists on disk must contain at least one
    ``def test_`` function definition. Catches the drift where
    a row points to a real file that is no longer a real test
    (e.g. renamed to a fixtures module)."""
    rows = _extract_5pent_rows()
    bad: list[str] = []
    for row in rows:
        for c in _extract_backtick_targets(row["tests"]):
            file_part = c.split("::", 1)[0]
            p = ROOT / file_part
            if not p.is_file():
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:
                bad.append(f"{file_part}: syntax error")
                continue
            has_test = any(
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                for node in ast.walk(tree)
            )
            if not has_test:
                bad.append(f"{file_part}: no def test_*")
    assert not bad, "§5-pent test files lack test_*:\n  " + "\n  ".join(bad)


def test_5pent_named_symbols_exist_where_specified():
    """For rows that name a specific symbol in parentheses
    (e.g. ``(`puct_score`, `select_action`)``), at least one of
    those symbols must be defined in the first linked
    implementation file. We are intentionally lenient (one of
    the symbols suffices) because a row may legitimately list
    multiple sub-symbols of which only the canonical one is a
    top-level def."""
    rows = _extract_5pent_rows()
    bad: list[str] = []
    for row in rows:
        impl_files = _extract_md_link_targets(row["impl"])
        symbols = _extract_backtick_symbols(row["impl"])
        if not impl_files or not symbols:
            continue
        first = ROOT / impl_files[0]
        if not first.is_file():
            continue  # already covered by previous test
        try:
            tree = ast.parse(first.read_text(encoding="utf-8"))
        except SyntaxError:
            bad.append(f"{impl_files[0]}: syntax error")
            continue
        defined: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            # Also accept top-level assignments / type aliases (e.g.
            # ``NuConfigMode = Literal["nu4", ...]``) and ``Final``-
            # typed module constants.
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgts:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
        # Also accept attribute-style symbols (``ClassName.attr``);
        # check the leftmost component is defined.
        flat_syms = {s.split(".", 1)[0] for s in symbols}
        # CLI-flag-style tokens (e.g. ``--table17``) are not symbols;
        # filter them out before strictness.
        flat_syms = {s for s in flat_syms if not s.startswith("-")
                     and not s.endswith(".py") and not s.endswith(".json")}
        if not flat_syms:
            continue
        if not (flat_syms & defined):
            bad.append(
                f"{row['section']} ({impl_files[0]}): "
                f"none of {flat_syms} found in {sorted(defined)[:8]}..."
            )
    assert not bad, "§5-pent symbol references not found:\n  " + "\n  ".join(bad)


def test_5pent_includes_d7_q14_fix_row():
    """The §5-pent table must include the Q14 garbage-math fallback
    row that points to ``cts/eval/garbage_filter.py`` (D-7 morning
    fix). A regression that drops this row would conceal the
    AIME-incident fix from reviewers."""
    rows = _extract_5pent_rows()
    q14 = [r for r in rows if "garbage_filter.py" in r["impl"]]
    assert len(q14) >= 1, "§5-pent missing the Q14 garbage_filter row"
    assert "is_garbage_math" in q14[0]["impl"], (
        "§5-pent Q14 row should name is_garbage_math symbol"
    )
    assert "test_dispatcher_fallback_mock.py" in q14[0]["tests"], (
        "§5-pent Q14 row should reference test_dispatcher_fallback_mock.py"
    )


def test_5pent_covers_core_paper_sections():
    """The §5-pent table must cover all primary paper sections
    (§3 DEQ, §4 method, §5 meta-policy, §6 training, §7 results)
    plus at least 4 appendices. A regression that drops coverage
    of §3 / §6 / §7 would silently shrink reviewer-checkable
    surface."""
    rows = _extract_5pent_rows()
    sections = {r["section"] for r in rows}
    must_have_prefixes = ["§3", "§4", "§5", "§6", "§7", "App."]
    missing = [
        p for p in must_have_prefixes
        if not any(s.startswith(p) for s in sections)
    ]
    assert not missing, f"§5-pent missing core section prefixes: {missing}"
    appendix_count = sum(1 for s in sections if s.startswith("App."))
    assert appendix_count >= 4, (
        f"§5-pent has only {appendix_count} appendix rows; expected >= 4"
    )
