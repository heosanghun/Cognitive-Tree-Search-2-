"""Torch-free static validation of every D-7 patch site.

These tests exist because the author's single development host
hits a kernel-level deadlock on ``import torch`` (REVIEWER_FAQ Q15);
running the existing pytest regression suite (which relies on torch)
is not always possible in the final week before the May 6 deadline.

Every test in this file:

- Runs in <100 ms
- Does not import torch / transformers / numpy / any heavy dependency
- Verifies the *presence and shape* of a D-7 patch by AST or regex
  inspection of the source file
- Is therefore safe to run in CI on a CPU-only runner with no GPU

The torch-dependent regressions live in
``tests/test_aime_garbage_fix.py`` and
``tests/test_pipeline_partial_save.py``; this file is *additional
coverage*, not a replacement.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _src(rel: str) -> str:
    p = ROOT / rel
    assert p.is_file(), f"{rel} missing"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Q14: AIME garbage two-layer fix
# ---------------------------------------------------------------------------


def test_q14_decode_from_z_star_has_problem_text_kwonly():
    """Fix #1 (paper-faithful): decode_from_z_star must accept the
    optional ``problem_text`` kw-only argument so the soft-prompt
    prefix is composed with the actual problem context, matching
    paper §4.3."""
    src = _src("cts/backbone/gemma_adapter.py")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "decode_from_z_star":
            kwonly_names = [a.arg for a in node.args.kwonlyargs]
            assert "problem_text" in kwonly_names, (
                f"decode_from_z_star kw-only args missing 'problem_text': {kwonly_names}"
            )
            found = True
            break
    assert found, "decode_from_z_star not found in gemma_adapter.py"


def test_q14_cts_episode_forwards_prompt():
    """cts_episode must pass ``problem_text=prompt`` to
    decode_from_z_star so every CTS / DEQ-only episode feeds both
    the latent and textual context to the decoder."""
    src = _src("cts/mcts/cts_episode.py")
    assert re.search(r"decode_from_z_star\([^)]*problem_text\s*=\s*prompt", src, re.DOTALL), (
        "cts_episode does not forward problem_text=prompt to decode_from_z_star"
    )


def test_q14_dispatcher_garbage_math_fallback():
    """Fix #2 (defence-in-depth): dispatcher must detect non-numeric
    extracted predictions on math benchmarks and fall back to
    greedy. Since the D-7 evening cleanup the predicate lives in
    ``cts/eval/garbage_filter.py::is_garbage_math``; the dispatcher
    imports + calls it instead of duplicating the logic inline."""
    src = _src("scripts/run_cts_eval_full.py")
    assert "fallback_prompt" in src, "fallback path marker missing"
    assert "from cts.eval.garbage_filter import is_garbage_math" in src, (
        "dispatcher must import is_garbage_math helper"
    )
    assert "is_garbage_math(benchmark," in src, (
        "dispatcher must call is_garbage_math(benchmark, ...)"
    )
    # The math-benchmark whitelist now lives in the helper module.
    helper = _src("cts/eval/garbage_filter.py")
    for bench in ('"math500"', '"aime"', '"aime_90"', '"gsm8k"'):
        assert bench in helper, f"math benchmark {bench} not in helper guard"


# ---------------------------------------------------------------------------
# Partial-save patch (commit ca3d601)
# ---------------------------------------------------------------------------


def test_partial_save_path_present():
    """run_cts_eval_full must write a partial-save snapshot at the
    canonical path so timeouts no longer destroy multi-hour work."""
    src = _src("scripts/run_cts_eval_full.py")
    assert "table2_results.partial.json" in src, (
        "partial-save snapshot path missing"
    )


def test_partial_save_helper_exists():
    src = _src("scripts/run_cts_eval_full.py")
    assert "def _flush_partial(" in src, "_flush_partial helper missing"


def test_partial_save_called_after_each_cell():
    """The flush must be called after every cell, not only at the
    end. Pattern: append(acc) then immediately _flush_partial()."""
    src = _src("scripts/run_cts_eval_full.py")
    assert re.search(r"append\(acc\)\s*\n\s*_flush_partial\(\)", src), (
        "_flush_partial() is not called immediately after acc append"
    )


# ---------------------------------------------------------------------------
# Pipeline --table*-limit knobs (commit ca3d601)
# ---------------------------------------------------------------------------


def test_pipeline_table2_limit_knob():
    src = _src("scripts/run_post_stage2_pipeline.py")
    assert "--table2-limit" in src
    assert "table2_limit" in src or "table2-limit" in src


def test_pipeline_table17_limit_knob():
    src = _src("scripts/run_post_stage2_pipeline.py")
    assert "--table17-limit" in src


# ---------------------------------------------------------------------------
# Watcher D-7 forwarding (commit 07fb924)
# ---------------------------------------------------------------------------


def test_watcher_forwards_d7_knobs():
    src = _src("scripts/wait_and_run_pipeline.ps1")
    for marker in ("$Table2Limit", "$Table17Limit", "$SkipVerify",
                   "--table2-limit", "--table17-limit", "--skip-verify"):
        assert marker in src, f"watcher missing marker: {marker}"
    # Default-0 guards preserve paper-faithful behavior when invoked
    # without the new flags.
    assert "$Table2Limit -gt 0" in src
    assert "$Table17Limit -gt 0" in src


# ---------------------------------------------------------------------------
# Q15: single-host blocker disclosure (commit 1a82a5d)
# ---------------------------------------------------------------------------


def test_q15_reviewer_faq_present():
    """REVIEWER_FAQ.md must include Q15 explaining why
    PAPER_VS_LOCAL.md still shows pre-retrain numbers."""
    faq = _src("REVIEWER_FAQ.md")
    assert "Q15" in faq, "REVIEWER_FAQ.md missing Q15"
    assert "single-host" in faq.lower() or "deadlock" in faq.lower(), (
        "REVIEWER_FAQ Q15 must explain the single-host kernel deadlock"
    )
    # Q15 must also disclose what we do NOT claim.
    assert "do **not** claim" in faq or "does NOT claim" in faq or \
           "not claim" in faq.lower(), (
        "REVIEWER_FAQ Q15 missing explicit non-claim disclosure"
    )


def test_q15_changelog_afternoon_entry():
    chlog = _src("CHANGELOG.md")
    assert "D-7 Apr 29 (afternoon)" in chlog or "single-host" in chlog, (
        "CHANGELOG missing D-7 Apr 29 afternoon Q15 entry"
    )


def test_q15_paper_vs_local_status_banner_updated():
    """The PAPER_VS_LOCAL.md status banner must point at Q15 so any
    reviewer who sees the pre-patch numbers knows where to find the
    full incident write-up."""
    pvl = _src("results/table2/PAPER_VS_LOCAL.md")
    # Either the banner mentions Q15 directly or it explains the
    # single-host blocker inline.
    assert "Q15" in pvl or "single-host" in pvl or "deadlock" in pvl.lower(), (
        "PAPER_VS_LOCAL.md status banner missing Q15 cross-reference"
    )


def test_q15_reproducibility_5_quat_present():
    rep = _src("REPRODUCIBILITY.md")
    assert "5-quat" in rep, "REPRODUCIBILITY missing §5-quat single-host blocker section"
    assert "Q15" in rep, "REPRODUCIBILITY §5-quat must cross-reference REVIEWER_FAQ Q15"


# ---------------------------------------------------------------------------
# Anonymous ZIP hardening (PROGRESS_REPORT exclusion, etc.)
# ---------------------------------------------------------------------------


def test_anon_zip_excludes_progress_report():
    """Author-facing PROGRESS_REPORT_*.md files must never leak into
    the public anonymous-submission ZIP."""
    src = _src("scripts/make_anonymous_submission.py")
    assert "PROGRESS_REPORT_*.md" in src, (
        "make_anonymous_submission.py missing PROGRESS_REPORT_*.md exclusion"
    )


def test_anon_zip_excludes_all_author_drafts():
    """Every author-draft pattern that lives gitignored on the
    author's disk must also be excluded from the ZIP build, so a
    *committed* draft (someone removes the gitignore line by
    mistake) still cannot leak into the double-blind submission."""
    src = _src("scripts/make_anonymous_submission.py")
    must_have = [
        "PROGRESS_REPORT_*.md",
        "OPENREVIEW_RESPONSE_PREP.md",
        "NEXT_TASKS_*.md",
        "PAPER_VS_LOCAL_FINAL.md",
        "PAPER_VS_LOCAL_INTUITIVE.md",
        "PAPER_CONSISTENCY_AUDIT.md",
        "ROOT_CAUSE_ANALYSIS.md",
        "EXPERIMENTAL_RESULTS.md",
    ]
    missing = [p for p in must_have if p not in src]
    assert not missing, (
        f"make_anonymous_submission.py EXCLUDE_GLOBS missing patterns: "
        f"{missing}"
    )


def test_ci_workflow_has_d7_assertions():
    """The GitHub Actions CI workflow must hard-fail on logs/ or
    PROGRESS_REPORT* leaks AND must verify D-7 fix content markers
    in the ZIP."""
    src = _src(".github/workflows/tests.yml")
    assert "logs leaked into ZIP" in src or "logs/" in src, (
        "CI workflow missing logs leak guard"
    )
    assert "PROGRESS_REPORT" in src, "CI workflow missing PROGRESS_REPORT guard"
    assert "test_aime_garbage_fix.py" in src, "CI workflow missing AIME garbage marker"
    assert "test_pipeline_partial_save.py" in src, (
        "CI workflow missing partial-save marker"
    )


# ---------------------------------------------------------------------------
# D12 final sanity script presence
# ---------------------------------------------------------------------------


def test_d12_final_check_script_present():
    """The D12 final sanity check script must exist and be
    importable as a module."""
    p = ROOT / "scripts/_d12_final_check.py"
    assert p.is_file(), "scripts/_d12_final_check.py missing"
    src = p.read_text(encoding="utf-8")
    assert "def main(" in src, "_d12_final_check.py missing main()"
    assert "section_d7_patches" in src, "_d12_final_check.py missing D-7 section"
    assert "section_docs" in src, "_d12_final_check.py missing docs section"
    assert "section_zip" in src, "_d12_final_check.py missing ZIP section"
    assert "section_reviewer_audit" in src, (
        "_d12_final_check.py missing Section 4 (reviewer-side audit)"
    )


# ---------------------------------------------------------------------------
# Plan C additions: LIMITATIONS.md + reviewer-side audit
# ---------------------------------------------------------------------------


def test_limitations_md_present():
    """LIMITATIONS.md is the consolidated reviewer-facing doc that
    points to every honest limitation in one place. Must enumerate
    the seven major limitations and the implementation-status
    table."""
    src = _src("LIMITATIONS.md")
    must_have = [
        "Compute-scaling gap",
        "Native Think baselines",
        "ARC-AGI-Text proxy",
        "CTS-2",  # (CTS-2 nu)
        "Coconut",
        "AIME garbage",
        "Single-host CUDA driver deadlock",
        "Reproducibility checklist coverage",
    ]
    missing = [m for m in must_have if m not in src]
    assert not missing, f"LIMITATIONS.md missing sections: {missing}"


def test_reviewer_local_audit_script_present():
    """scripts/_reviewer_local_audit.py is the reviewer-side static
    audit (distinct from author's _d12_final_check.py). Must have
    four sections (headline claims / patches / repro / docs) and
    must not import torch."""
    src = _src("scripts/_reviewer_local_audit.py")
    assert "section_headline_claims" in src
    assert "section_d7_patch_sites" in src
    assert "section_reproducibility" in src
    assert "section_docs" in src
    # Reviewer audit MUST be torch-free so a reviewer without GPU
    # can still verify the static surface. We check the actual AST
    # import statements (not docstring mentions).
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad.extend(a.name for a in node.names if a.name.split(".")[0]
                       in {"torch", "transformers", "numpy"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"torch", "transformers", "numpy"}:
                bad.append(node.module)
    assert not bad, (
        f"reviewer audit must not import torch/transformers/numpy "
        f"(Q15 disclosure principle); offenders: {bad}"
    )


def test_garbage_filter_helper_present():
    """Q14 Fix B is factored out into cts/eval/garbage_filter.py
    (rather than duplicated inline in run_cts_eval_full.py).
    Must export ``is_garbage_math``, ``is_math_benchmark``, and
    ``MATH_BENCHMARKS`` and must not import torch."""
    src = _src("cts/eval/garbage_filter.py")
    assert "def is_garbage_math(" in src
    assert "def is_math_benchmark(" in src
    assert "MATH_BENCHMARKS" in src
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad.extend(a.name for a in node.names if a.name.split(".")[0]
                       in {"torch", "transformers", "numpy"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"torch", "transformers", "numpy"}:
                bad.append(node.module)
    assert not bad, f"garbage_filter must be torch-free; offenders: {bad}"


def test_dispatcher_imports_garbage_filter_helper():
    """The dispatcher must IMPORT the helper rather than
    duplicate the predicate inline (defence against the predicate
    drifting out of sync between the CTS and DEQ-only paths)."""
    src = _src("scripts/run_cts_eval_full.py")
    assert "from cts.eval.garbage_filter import is_garbage_math" in src, (
        "run_cts_eval_full.py must import is_garbage_math from helper"
    )
    # The inline duplicate predicate must be GONE.
    assert "_is_garbage_math = (" not in src, (
        "run_cts_eval_full.py still has the inline _is_garbage_math "
        "tuple form; should call helper instead"
    )


def test_dispatcher_mock_test_present():
    """The mock-based behavioural test for the dispatcher
    fallback must exist and be torch-free (loads garbage_filter
    via importlib to bypass cts/__init__.py)."""
    src = _src("tests/test_dispatcher_fallback_mock.py")
    assert "_simulate_dispatcher_decision" in src
    assert "test_dispatcher_falls_back_on_canonical_aime_garbage" in src
    assert "importlib.util" in src, (
        "mock test must use importlib to bypass cts package init"
    )


def test_replication_shell_script_present():
    """The reviewer-facing one-command replication script must
    exist, be executable in spirit (bash shebang), forward the
    D-7 limit knobs, and offer a --static-only mode for
    reviewers without GPU access."""
    src = _src("scripts/replicate_neurips_2026.sh")
    assert src.startswith("#!/usr/bin/env bash")
    for marker in ("--full", "--static-only", "--table2-limit",
                   "--table17-limit", "_reviewer_local_audit.py",
                   "test_d7_static_validation.py",
                   "test_dispatcher_fallback_mock.py",
                   "run_post_stage2_pipeline.py"):
        assert marker in src, f"replicate script missing marker: {marker}"


def test_paper_code_mapping_table_validator_present():
    """The §5-pent table validator (parses the markdown table and
    asserts every linked file/symbol/test exists) must exist and
    be torch-free."""
    src = _src("tests/test_paper_code_mapping_table.py")
    assert "_extract_5pent_rows" in src
    assert "test_5pent_table_is_non_empty" in src
    assert "test_5pent_every_impl_file_exists" in src
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad.extend(a.name for a in node.names if a.name.split(".")[0]
                       in {"torch", "transformers", "numpy"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"torch", "transformers", "numpy"}:
                bad.append(node.module)
    assert not bad, f"5-pent validator must be torch-free; offenders: {bad}"


def test_stage2_training_meta_contract_test_present():
    """The Stage 2 training_meta writer/reader contract test must
    exist and exercise both sides (writer + reader) without
    importing torch."""
    src = _src("tests/test_stage2_training_meta_static.py")
    assert "REQUIRED_TRAINING_META_KEYS" in src
    assert "test_writer_persists_every_required_key" in src
    assert "test_reader_loads_training_meta_block" in src
    assert "test_writer_reader_keys_are_consistent" in src
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad.extend(a.name for a in node.names if a.name.split(".")[0]
                       in {"torch", "transformers", "numpy"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"torch", "transformers", "numpy"}:
                bad.append(node.module)
    assert not bad, f"training_meta test must be torch-free; offenders: {bad}"


def test_run_cts_eval_full_has_dry_run_mode():
    """``scripts/run_cts_eval_full.py`` must expose a --dry-run
    flag that returns 0 without importing torch (REVIEWER_FAQ
    Q15: torch-free verification of the eval matrix)."""
    src = _src("scripts/run_cts_eval_full.py")
    assert '"--dry-run"' in src, "missing --dry-run argparse argument"
    assert "_dry_run(args)" in src, "main() must dispatch to _dry_run on --dry-run"
    assert "def _dry_run(args:" in src or "def _dry_run(" in src, (
        "_dry_run helper must be defined"
    )
    # The helper's docstring must promise no-torch.
    pat = re.search(r'def _dry_run\([^)]*\)[^:]*:\s*"""(.*?)"""', src, re.DOTALL)
    assert pat is not None
    assert "torch" in pat.group(1).lower() and "no" in pat.group(1).lower(), (
        "_dry_run docstring must promise no-torch behaviour"
    )


def test_readme_has_reviewer_quick_start():
    """README.md must surface the reviewer quick-start commands
    (audit script + replication script + dry-run preview) in a
    dedicated section so a 30-second skim reaches them."""
    src = _src("README.md")
    assert "Reviewer Quick Start" in src
    must_have = [
        "_reviewer_local_audit.py",
        "replicate_neurips_2026.sh",
        "--dry-run",
        "LIMITATIONS.md",
        "REVIEWER_FAQ.md",
        "5-pent",
    ]
    missing = [m for m in must_have if m not in src]
    assert not missing, f"README Reviewer Quick Start missing markers: {missing}"


def test_reproducibility_5_pent_extended_mapping():
    """REPRODUCIBILITY 5-pent must contain the extended paper-claim
    -> code-line mapping table (covering core paper sections, not
    just the P0 audit fixes that 5-bis covers)."""
    src = _src("REPRODUCIBILITY.md")
    assert "5-pent" in src
    assert "Paper claim" in src
    # Spot-check a representative subset of paper sections.
    for marker in ("§3.1", "§4.1", "§4.5", "§6", "§7.1", "§7.5",
                   "garbage_filter.py", "meta_policy.py"):
        assert marker in src, f"5-pent table missing marker: {marker}"


def test_anon_zip_byte_invariants_test_present():
    """Third leak-defence layer: byte-level invariants on the
    anonymous ZIP (path patterns + content tokens + structural
    invariants). Must be torch-free."""
    src = _src("tests/test_anon_zip_byte_invariants.py")
    for marker in ("EXPECTED_ENTRY_POINTS", "AUTHOR_DRAFT_HEADERS",
                   "MAX_FILE_BYTES", "ALLOWED_DOT_PREFIXES"):
        assert marker in src, f"byte-invariants test missing {marker}"
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad.extend(a.name for a in node.names if a.name.split(".")[0]
                       in {"torch", "transformers", "numpy"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"torch", "transformers", "numpy"}:
                bad.append(node.module)
    assert not bad, f"byte-invariants test must be torch-free; offenders: {bad}"


def test_paper_section_alignment_test_present():
    """Cross-document §-numbering alignment test must exist and
    cover every reviewer-facing markdown."""
    src = _src("tests/test_paper_section_alignment.py")
    for marker in ("test_5pent_covers_every_primary_section_family",
                   "test_faq_section_refs_subset_of_reproducibility",
                   "test_no_orphaned_section_in_paper_vs_local"):
        assert marker in src, f"section-alignment test missing {marker}"


def test_reviewer_walkthrough_script_present():
    """Reviewer walkthrough must exist, be runnable as a plain
    script, AND be runnable as a notebook (cell markers)."""
    src = _src("scripts/reviewer_walkthrough.py")
    assert "# %%" in src, "walkthrough must have notebook cell markers"
    assert "_extract_5pent_rows" in src
    assert "cell_drill_q14_fix" in src
    assert "if __name__" in src


def test_d12_final_check_has_export_verdict_mode():
    """The author's D12 sanity script must support
    ``--export-verdict PATH`` (writes JSON + paste-ready
    markdown) and ``--quiet``."""
    src = _src("scripts/_d12_final_check.py")
    assert "--export-verdict" in src
    assert "--quiet" in src
    assert "_build_verdict_payload" in src
    assert "_emit_paste_ready_summary" in src
    assert "schema" in src and "cts_neurips2026.d12_sanity.v1" in src, (
        "verdict JSON must carry a schema version for future migration"
    )


def test_replicate_script_has_ci_mode():
    """The reviewer replication script must support a --ci-mode
    flag (no GPU, structured exit codes, exports D12 verdict
    artefact for GitHub Actions matrix)."""
    src = _src("scripts/replicate_neurips_2026.sh")
    assert "--ci-mode" in src
    assert 'MODE="ci-mode"' in src
    assert "--export-verdict" in src, (
        "ci-mode must call _d12_final_check.py with --export-verdict"
    )


def test_reviewer_faq_q16_ci_guide_present():
    """REVIEWER_FAQ Q16 must explain how to run the verification
    suite without GitHub Actions / admin rights / GPU."""
    src = _src("REVIEWER_FAQ.md")
    assert "Q16." in src
    for marker in ("--static-only", "--export-verdict", "--ci-mode",
                   "GitHub Actions"):
        assert marker in src, f"Q16 missing marker: {marker}"


def test_reviewer_faq_q17_crossref_present():
    """REVIEWER_FAQ Q17 must give the reviewer a one-command
    path from a paper §-number to a runnable test."""
    src = _src("REVIEWER_FAQ.md")
    assert "Q17." in src
    for marker in ("reviewer_walkthrough.py", "5-pent",
                   "test_paper_code_mapping_table",
                   "test_paper_section_alignment"):
        assert marker in src, f"Q17 missing marker: {marker}"


def test_walkthrough_invariants_test_present():
    """The walkthrough is the reviewer's interactive entry point;
    its zero-MISS invariant must be locked by a regression test."""
    src = _src("tests/test_reviewer_walkthrough_invariants.py")
    for marker in ("test_walkthrough_zero_miss",
                   "test_walkthrough_row_count_matches_5pent",
                   "test_walkthrough_drills_q14_and_meta_policy"):
        assert marker in src, f"walkthrough invariants test missing {marker}"


def test_changelog_d7_completeness_test_present():
    """The CHANGELOG must self-validate: every D-7 batch is
    required to cite §-section / Q-number / Plan letter."""
    src = _src("tests/test_changelog_d7_completeness.py")
    for marker in ("test_every_d7_block_cites_a_section_or_qnumber",
                   "test_every_known_plan_block_cites_required_artefacts",
                   "test_d7_test_suite_additions_cite_pass_ratio"):
        assert marker in src, f"CHANGELOG completeness test missing {marker}"


def test_d12_section_5_byte_invariants_integrated():
    """Section 5 of the D12 sanity script must run the byte-
    invariants suite in-process so the gate fails on
    structural ZIP corruption even if the path-pattern audit
    passes."""
    src = _src("scripts/_d12_final_check.py")
    assert "section_byte_invariants" in src
    assert "Section 5" in src
    assert "byte-invariants" in src or "byte invariants" in src.lower()


def test_ci_workflow_uploads_d12_verdict_artefact():
    """The GitHub Actions workflow must export the D12 verdict
    so an external reviewer can download the JSON + paste-
    ready markdown without running the script themselves."""
    src = _src(".github/workflows/tests.yml")
    assert "Upload D12 verdict artefact" in src
    assert "actions/upload-artifact" in src
    assert "results/d12_verdict.json" in src
    assert "results/d12_verdict.md" in src
    assert "Replication script CI mode" in src
    assert "--ci-mode" in src


def test_reviewer_local_audit_has_json_mode():
    """The reviewer audit must support --json so a reviewer
    can program-attach the structured verdict to their
    review template (or pipe to ``jq``)."""
    src = _src("scripts/_reviewer_local_audit.py")
    assert "--json" in src
    assert "_build_json_payload" in src
    assert "schema" in src and "cts_neurips2026.reviewer_audit.v1" in src
    # Must be torch-free (still).
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad.extend(a.name for a in node.names if a.name.split(".")[0]
                       in {"torch", "transformers", "numpy"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in {"torch", "transformers", "numpy"}:
                bad.append(node.module)
    assert not bad, f"reviewer audit must remain torch-free; offenders: {bad}"


def test_walkthrough_has_html_mode():
    """The walkthrough must support --html so a reviewer can
    open the navigation in a browser without having a
    terminal."""
    src = _src("scripts/reviewer_walkthrough.py")
    assert "--html" in src
    assert "_render_html" in src
    assert "<!DOCTYPE html>" in src
    assert "Verdict:" in src or "verdict" in src.lower()


def test_reproducibility_checklist_coverage_test_present():
    """REPRODUCIBILITY.md must be statically validated for
    NeurIPS 2026 Reproducibility Checklist coverage (13
    sections + 4 extension sections + §5 link integrity)."""
    src = _src("tests/test_reproducibility_checklist_coverage.py")
    for marker in ("test_all_13_checklist_sections_present",
                   "test_every_section_has_substantive_body",
                   "test_all_four_extension_sections_present",
                   "test_5_pent_table_has_the_expected_minimum_row_count"):
        assert marker in src, f"checklist coverage test missing {marker}"


def test_limitations_completeness_test_present():
    """LIMITATIONS.md must be statically validated for the
    structured (Limitation / Mitigation / Non-claim /
    Crossref) contract on every numbered section."""
    src = _src("tests/test_limitations_completeness.py")
    for marker in ("test_every_limitation_has_what_done_subsection",
                   "test_every_limitation_has_explicit_non_claim_disclosure",
                   "test_every_limitation_has_crossreference",
                   "test_q14_and_q15_limitations_present",
                   "test_no_internal_filenames_leak"):
        assert marker in src, f"LIMITATIONS completeness test missing {marker}"


def test_pre_submission_orchestrator_present():
    """The D12 GO/NO-GO orchestrator must exist and chain
    every static gate (10 suites + reviewer audit + walk-
    through + d12 sanity + byte invariants + replication)."""
    src = _src("scripts/run_pre_submission_audit.py")
    for marker in ("step_static_suites", "step_reviewer_audit",
                   "step_walkthrough", "step_d12_sanity",
                   "step_replication_ci_mode",
                   "GO: every gate passed",
                   "NO-GO"):
        assert marker in src, f"orchestrator missing {marker}"


def test_gitignore_excludes_author_drafts():
    """Author-facing drafts (PROGRESS_REPORT_*.md,
    OPENREVIEW_RESPONSE_PREP.md, NEXT_TASKS_*.md, etc.) must
    never leak into the public repo."""
    src = _src(".gitignore")
    for pat in ("PROGRESS_REPORT_*.md", "OPENREVIEW_RESPONSE_PREP.md",
                "NEXT_TASKS_*.md", "PAPER_VS_LOCAL_FINAL.md"):
        assert pat in src, f".gitignore missing author-draft pattern: {pat}"
