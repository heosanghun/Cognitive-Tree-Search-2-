#!/usr/bin/env python3
"""Reviewer walkthrough: navigate paper claims to source code in
order, with one cell per paper section.

Two execution modes:

1. As a plain script (no torch, ~1 second):
       python scripts/reviewer_walkthrough.py
   Prints every paper section's claim, the implementation file
   the claim resolves to, and the regression-test path that
   covers it. Walks the entire §5-pent mapping table.

2. As a VS Code / Jupyter notebook (cell markers ``# %%``):
       Open in any Jupyter-compatible IDE (e.g. VS Code with
       the Python extension), click "Run Cell" at each marker.
       Each cell prints the claim and shows the file with line
       numbers; the next cell shows the test that covers the claim.

The walkthrough is **deliberately torch-free** so a reviewer
without GPU can run it on any CPython 3.10+ environment. It
parses the live REPRODUCIBILITY.md §5-pent table, so any drift
in the table is reflected here automatically (see
``tests/test_paper_code_mapping_table.py`` for the contract).
"""

# %%
from __future__ import annotations

import argparse
import html as _html
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "REPRODUCIBILITY.md"


# %%
# ----- Helpers (no torch) ---------------------------------------------------


def _extract_5pent_rows() -> list[dict[str, str]]:
    src = REPRO.read_text(encoding="utf-8")
    m = re.search(r"###\s*5-pent\..*?\n(.*?)(?=\n##\s|\Z)", src, re.DOTALL)
    if m is None:
        print("ERROR: REPRODUCIBILITY.md missing §5-pent section.")
        sys.exit(2)
    body = m.group(1)
    rows: list[dict[str, str]] = []
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
        rows.append({
            "section": cells[0],
            "claim": cells[1],
            "impl": cells[2],
            "tests": cells[3],
        })
    return rows


def _md_links(cell: str) -> list[str]:
    return re.findall(r"\(([^)]+\.[a-zA-Z0-9]+)\)", cell)


def _backtick_paths(cell: str) -> list[str]:
    return re.findall(r"`([^`]+?\.py(?:::[A-Za-z_][A-Za-z0-9_]*)?)`", cell)


def _show_file_head(p: Path, n_lines: int = 12) -> str:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        return f"(could not read {p}: {exc})"
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines[:n_lines], start=1):
        out.append(f"  {i:4d} | {line}")
    if len(lines) > n_lines:
        out.append(f"       ... ({len(lines) - n_lines} more lines)")
    return "\n".join(out)


def _exists(rel: str) -> bool:
    return (ROOT / rel.split("::", 1)[0]).is_file()


# %%
# ----- Cell: header --------------------------------------------------------


def cell_header() -> None:
    print("=" * 78)
    print("CTS NeurIPS 2026 - Reviewer Walkthrough")
    print("=" * 78)
    print(textwrap.dedent("""
        This walkthrough takes you through every primary claim in the
        paper and shows the source file + regression test that backs it.
        Each section below corresponds to one row in REPRODUCIBILITY.md
        section 5-pent.

        Conventions used below:
          [OK]   - the linked file/test resolves on disk
          [MISS] - the linked file/test is missing (file a review comment)

        Total walking time: ~1 second; no torch / GPU needed.
    """).strip())
    print()


# %%
# ----- Cell: enumerate paper sections in order -----------------------------


def cell_enumerate_paper_sections(rows: list[dict[str, str]]) -> None:
    print("-" * 78)
    print(f"{len(rows)} paper-claim rows mapped to source code")
    print("-" * 78)
    by_section: dict[str, list[str]] = {}
    for r in rows:
        by_section.setdefault(r["section"], []).append(r["claim"])
    for sec in sorted(by_section.keys(), key=_section_sort_key):
        print(f"\n  {sec}")
        for c in by_section[sec]:
            print(f"    - {c[:78]}")
    print()


def _section_sort_key(s: str) -> tuple:
    # Paper sections (§3.1, §4.5, §7.6 (Table 19), App. C, ...)
    m = re.match(r"§(\d+)(?:\.(\d+))?", s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) else 0
        return (0, major, minor, s)
    if s.startswith("App."):
        return (1, ord(s[5]) if len(s) > 5 else 0, 0, s)
    if s == "Q14 fix":
        return (2, 0, 0, s)
    return (3, 0, 0, s)


# %%
# ----- Cell: resolve and verify each row -----------------------------------


def cell_resolve_each_row(rows: list[dict[str, str]]) -> tuple[int, int]:
    print("-" * 78)
    print("Per-row resolution (impl files + test files)")
    print("-" * 78)
    ok_count = 0
    miss_count = 0
    for r in rows:
        impls = _md_links(r["impl"])
        tests = _backtick_paths(r["tests"])
        any_test_exists = any(_exists(t) for t in tests)
        all_impls_exist = all((ROOT / p).exists() for p in impls)
        flag = "OK  " if (all_impls_exist and any_test_exists) else "MISS"
        if all_impls_exist and any_test_exists:
            ok_count += 1
        else:
            miss_count += 1
        print(f"\n  [{flag}] {r['section']}: {r['claim'][:60]}")
        for p in impls:
            mark = "[OK]" if (ROOT / p).exists() else "[MISS]"
            print(f"        impl  {mark} {p}")
        for t in tests:
            mark = "[OK]" if _exists(t) else "[MISS]"
            print(f"        test  {mark} {t}")
    print()
    return ok_count, miss_count


# %%
# ----- Cell: drill into a specific row (example: Q14 garbage-math fallback)


def cell_drill_q14_fix() -> None:
    print("-" * 78)
    print("Drill: Q14 garbage-math fallback (the AIME-incident fix)")
    print("-" * 78)
    print("\n[1/2] Implementation: cts/eval/garbage_filter.py (head)\n")
    print(_show_file_head(ROOT / "cts/eval/garbage_filter.py", n_lines=20))
    print("\n[2/2] Behavioural test: tests/test_dispatcher_fallback_mock.py (head)\n")
    print(_show_file_head(ROOT / "tests/test_dispatcher_fallback_mock.py", n_lines=12))
    print()


# %%
# ----- Cell: drill into the meta-policy (paper §4.5 nu-vector control) -----


def cell_drill_meta_policy() -> None:
    print("-" * 78)
    print("Drill: §4.5 nu-vector adaptive control")
    print("-" * 78)
    print("\nImplementation: cts/policy/meta_policy.py (head)\n")
    print(_show_file_head(ROOT / "cts/policy/meta_policy.py", n_lines=18))
    print()


# %%
# ----- Cell: footer with verdict -------------------------------------------


def cell_footer(ok_count: int, miss_count: int) -> int:
    print("=" * 78)
    print(f"Walkthrough verdict: {ok_count} OK, {miss_count} MISS")
    print("=" * 78)
    if miss_count == 0:
        print("\n>>> ALL GREEN: every paper claim resolves to a real file + test.")
        print(">>> Next step: run `bash scripts/replicate_neurips_2026.sh --static-only`")
        print(">>>            (~2 seconds) for the full reviewer audit.")
        return 0
    print(f"\n>>> {miss_count} rows could not be fully resolved. File a review")
    print(">>> comment quoting the [MISS] lines above; the author should patch.")
    return 1


# %%
# ----- HTML export ---------------------------------------------------------


_HTML_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     max-width:980px;margin:24px auto;padding:0 16px;color:#222;
     line-height:1.55;}
h1{border-bottom:2px solid #06c;padding-bottom:4px;}
h2{margin-top:36px;color:#06c;}
.row{border:1px solid #ddd;border-radius:6px;padding:10px 14px;
     margin:8px 0;background:#fafafa;}
.row .ok{color:#080;font-weight:bold;}
.row .miss{color:#c00;font-weight:bold;}
.row .section{display:inline-block;background:#06c;color:#fff;
              padding:2px 8px;border-radius:3px;font-size:13px;
              font-family:monospace;margin-right:8px;}
.row .claim{margin:6px 0;}
.row .files{font-family:monospace;font-size:13px;color:#444;
            margin-left:12px;}
.row .files a{color:#06c;text-decoration:none;}
.row .files a:hover{text-decoration:underline;}
.summary{padding:12px 16px;border-radius:6px;margin:24px 0;
         font-size:18px;font-weight:bold;}
.summary.green{background:#d4edda;color:#155724;border:1px solid #c3e6cb;}
.summary.red{background:#f8d7da;color:#721c24;border:1px solid #f5c6cb;}
pre{background:#1e1e1e;color:#dcdcdc;padding:12px;border-radius:6px;
    overflow-x:auto;font-size:13px;line-height:1.4;}
.timestamp{color:#888;font-size:13px;}
"""


def _render_html(rows: list[dict[str, str]], ok: int, miss: int) -> str:
    import datetime as _dt
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'><head><meta charset='utf-8'>")
    parts.append("<title>CTS NeurIPS 2026 - Reviewer Walkthrough</title>")
    parts.append(f"<style>{_HTML_CSS}</style></head><body>")
    parts.append("<h1>CTS NeurIPS 2026 - Reviewer Walkthrough</h1>")
    parts.append(
        "<p>Every primary paper claim mapped to a source file + "
        "regression test. Generated from the live <code>REPRODUCIBILITY.md "
        "&sect;5-pent</code> table; if any row drifts, CI blocks the merge "
        "(<code>tests/test_reviewer_walkthrough_invariants.py</code>).</p>"
    )
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    parts.append(f"<p class='timestamp'>Generated: {ts}</p>")

    klass = "green" if miss == 0 else "red"
    verdict = f"Verdict: {ok} OK, {miss} MISS"
    parts.append(f"<div class='summary {klass}'>{verdict}</div>")

    parts.append("<h2>Paper-claim mapping</h2>")
    for r in rows:
        impls = _md_links(r["impl"])
        tests = _backtick_paths(r["tests"])
        impl_ok = all((ROOT / p).exists() for p in impls)
        test_ok = any(_exists(t) for t in tests)
        flag_html = ("<span class='ok'>OK</span>" if (impl_ok and test_ok)
                     else "<span class='miss'>MISS</span>")
        parts.append("<div class='row'>")
        parts.append(
            f"<span class='section'>{_html.escape(r['section'])}</span>"
            f"{flag_html}"
        )
        parts.append(f"<div class='claim'>{_html.escape(r['claim'])}</div>")
        for p in impls:
            mark = "OK" if (ROOT / p).exists() else "MISS"
            parts.append(
                f"<div class='files'>impl &nbsp;[{mark}]&nbsp;"
                f"<a href='{_html.escape(p)}'>{_html.escape(p)}</a></div>"
            )
        for t in tests:
            mark = "OK" if _exists(t) else "MISS"
            parts.append(
                f"<div class='files'>test &nbsp;[{mark}]&nbsp;"
                f"<a href='{_html.escape(t.split('::')[0])}'>{_html.escape(t)}</a></div>"
            )
        parts.append("</div>")

    parts.append("<h2>Drill: Q14 garbage-math fallback</h2>")
    parts.append("<pre>")
    parts.append(_html.escape(_show_file_head(ROOT / "cts/eval/garbage_filter.py", n_lines=20)))
    parts.append("</pre>")

    parts.append("<h2>Drill: nu-vector adaptive control (paper &sect;4.5)</h2>")
    parts.append("<pre>")
    parts.append(_html.escape(_show_file_head(ROOT / "cts/policy/meta_policy.py", n_lines=18)))
    parts.append("</pre>")

    parts.append(
        "<p>Next step: <code>bash scripts/replicate_neurips_2026.sh "
        "--static-only</code> for the full reviewer audit (~2 seconds).</p>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


# %%
# ----- Driver ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reviewer walkthrough: navigate paper claims to source code"
    )
    parser.add_argument(
        "--html", metavar="PATH", default=None,
        help="Write a self-contained HTML version of the walkthrough to "
             "PATH (no JS, no external CSS, openable in any browser). "
             "When set, normal terminal output is suppressed.",
    )
    args = parser.parse_args(argv)

    rows = _extract_5pent_rows()
    if args.html:
        # Compute OK/MISS without printing.
        ok = miss = 0
        for r in rows:
            impls = _md_links(r["impl"])
            tests = _backtick_paths(r["tests"])
            impl_ok = all((ROOT / p).exists() for p in impls)
            test_ok = any(_exists(t) for t in tests)
            if impl_ok and test_ok:
                ok += 1
            else:
                miss += 1
        out_path = Path(args.html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_html(rows, ok, miss), encoding="utf-8")
        print(f"HTML walkthrough written to: {out_path}")
        print(f"Verdict: {ok} OK, {miss} MISS")
        return 0 if miss == 0 else 1

    cell_header()
    cell_enumerate_paper_sections(rows)
    ok, miss = cell_resolve_each_row(rows)
    cell_drill_q14_fix()
    cell_drill_meta_policy()
    return cell_footer(ok, miss)


if __name__ == "__main__":
    raise SystemExit(main())
