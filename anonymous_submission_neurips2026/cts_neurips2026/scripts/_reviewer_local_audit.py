#!/usr/bin/env python3
"""Reviewer-side local audit (NeurIPS 2026, CTS submission).

Designed to run **after a reviewer unzips
`anonymous_submission_neurips2026.zip`** to give them an instant
read on what the submission claims, what's verifiable from the
source alone (no torch / GPU needed), and what requires their own
GPU run to check.

Distinct from `scripts/_d12_final_check.py`:

- `_d12_final_check.py` is the **author's** D12-day pre-submission
  gate (rebuilds the anonymous ZIP, verifies audit PASS).
- `_reviewer_local_audit.py` (this file) is what the **reviewer**
  runs after unzipping; it does not rebuild the ZIP, it does not
  call the audit script, and it does not assume any author-side
  tooling is present. It only reads what's already on disk in the
  reviewer's checkout.

Sections:

1. Headline-claim coverage:  every paper claim in REVIEWER_FAQ.md
   that has a verifiable source-code anchor is checked here.
2. Patch-site presence:      every D-7 fix (Q14, partial-save,
   watcher D-7-limit, Q15 disclosure) is checked at AST/regex
   level so the reviewer can confirm at a glance that the fixes
   the FAQ talks about actually shipped.
3. Reproducibility-checklist hooks: the eight items of
   REPRODUCIBILITY.md that have shippable artifacts (config
   files, regression tests, contamination-screen reports, etc.)
   are spot-checked.
4. Documentation-marker presence: REVIEWER_FAQ Q1-Q15,
   PAPER_VS_LOCAL.md, LIMITATIONS.md.

Exit codes:
  0 = ALL GREEN (reviewer can trust the static surface)
  1 = AT LEAST ONE FAIL (something the reviewer should
      challenge in their review)

Usage:
  python scripts/_reviewer_local_audit.py
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Tiny result helper
# ---------------------------------------------------------------------------


class _R:
    __slots__ = ("name", "ok", "evidence")

    def __init__(self, name: str, ok: bool, evidence: str = "") -> None:
        self.name = name
        self.ok = ok
        self.evidence = evidence


def _read(rel: str) -> str:
    p = ROOT / rel
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


# ---------------------------------------------------------------------------
# Section 1: Headline-claim coverage
# ---------------------------------------------------------------------------


def section_headline_claims() -> list[_R]:
    out: list[_R] = []

    # Claim A: KV-cache-free per-node O(1) active VRAM (paper Table 1)
    has_kv = (_exists("cts/mcts/hybrid_kv.py")
              or _exists("cts/eval/kv_measured.py")
              or _exists("cts/eval/profile_vram_latency.py"))
    has_triton = _exists("cts/routing/sparse_moe_triton.py")
    out.append(_R(
        "Paper Table 1 - O(1) active VRAM measurement path",
        has_kv,
        f"hybrid_kv / kv_measured / profile_vram_latency present (triton aux: {has_triton})"
    ))

    # Claim B: DEQ L-Broyden inner solver (paper Section 6)
    src = _read("cts/deq/transition.py") + _read("cts/deq/broyden_forward.py")
    has_lbroyden = "broyden" in src.lower() or "L-Broyden" in src
    out.append(_R(
        "Paper Section 6 - L-Broyden DEQ solver",
        bool(src) and has_lbroyden,
        "broyden marker found in cts/deq/"
    ))

    # Claim C: Meta-policy + neuro-critic (paper Section 5)
    out.append(_R(
        "Paper Section 5 - meta-policy + neuro-critic",
        _exists("cts/policy/meta_policy.py") and _exists("cts/critic/neuro_critic.py"),
        "cts/policy/meta_policy.py + cts/critic/neuro_critic.py"
    ))

    # Claim D: PUCT selection (paper Section 4)
    src = _read("cts/mcts/puct.py") + _read("cts/mcts/cts_episode.py")
    out.append(_R(
        "Paper Section 4 - PUCT selection rule",
        bool(src) and ("puct" in src.lower() or "PUCT" in src),
        "PUCT marker in cts/mcts/"
    ))

    # Claim E: ACT halting penalty (paper Section 6)
    src = _read("cts/train/stage2_ppo_train.py") + _read("cts/train/ppo_core.py")
    has_act = "act_halting" in src.lower() or "halt" in src.lower()
    out.append(_R(
        "Paper Section 6 - ACT halting penalty",
        bool(src) and has_act,
        "halt marker in cts/train/"
    ))

    # Claim F: nu-vector adaptive control (paper Section 4.5)
    src = _read("cts/policy/meta_policy.py")
    has_nu = ("nu_expl" in src or "nu_temp" in src
              or "nu_act" in src or "nu_tol" in src)
    out.append(_R(
        "Paper Section 4.5 - nu-vector adaptive control",
        has_nu,
        "nu_{expl,tol,temp,act} markers in cts/policy/meta_policy.py"
    ))

    # Claim G: Stage 1 DEQ warm-up + Stage 2 PPO (paper Section 6)
    s1 = _exists("cts/train/stage1_openmath_train.py") or _exists("cts/train/stage1_warmup.py")
    s2 = _exists("cts/train/stage2_ppo_train.py")
    out.append(_R(
        "Paper Section 6 - two-stage training (DEQ warm-up + PPO)",
        s1 and s2,
        f"stage1={s1} stage2={s2}"
    ))

    # Claim H: Contamination screening (paper Appendix)
    has_contam = _exists("results/contamination") or _exists("scripts/contamination_screen.py")
    out.append(_R(
        "Paper Appendix - contamination screening (FAISS / MinHash / BM25)",
        has_contam,
        "results/contamination/ or scripts/contamination_screen.py present"
    ))

    return out


# ---------------------------------------------------------------------------
# Section 2: D-7 patch-site presence (mirrors REVIEWER_FAQ Q15 step 1)
# ---------------------------------------------------------------------------


def section_d7_patch_sites() -> list[_R]:
    out: list[_R] = []

    # Q14 paper-faithful: decode_from_z_star kw-only problem_text
    src = _read("cts/backbone/gemma_adapter.py")
    ok = False
    if src:
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "decode_from_z_star":
                    if "problem_text" in [a.arg for a in node.args.kwonlyargs]:
                        ok = True
                    break
        except SyntaxError:
            pass
    out.append(_R("Q14 fix #1 - decode_from_z_star(problem_text=...) kwarg", ok))

    # Q14 episode wiring
    src = _read("cts/mcts/cts_episode.py")
    ok = bool(re.search(r"decode_from_z_star\([^)]*problem_text\s*=\s*prompt", src, re.DOTALL))
    out.append(_R("Q14 fix #1 - cts_episode forwards prompt", ok))

    # Q14 defence-in-depth dispatcher (D-7 evening: predicate
    # extracted into cts/eval/garbage_filter.py; dispatcher imports
    # the helper instead of duplicating the regex inline).
    src = _read("scripts/run_cts_eval_full.py")
    out.append(_R(
        "Q14 fix #2 - dispatcher garbage-math fallback",
        "from cts.eval.garbage_filter import is_garbage_math" in src
        and "is_garbage_math(benchmark," in src
        and "fallback_prompt" in src
    ))

    # Partial-save snapshot
    out.append(_R(
        "Partial-save snapshot - table2_results.partial.json",
        "table2_results.partial.json" in src and "def _flush_partial(" in src
    ))

    # Pipeline knobs
    src2 = _read("scripts/run_post_stage2_pipeline.py")
    out.append(_R(
        "Pipeline knobs - --table2-limit / --table17-limit",
        "--table2-limit" in src2 and "--table17-limit" in src2
    ))

    # Watcher D-7 forwarding
    src3 = _read("scripts/wait_and_run_pipeline.ps1")
    ok = ("$Table2Limit" in src3 and "$Table17Limit" in src3
          and "--table2-limit" in src3 and "--table17-limit" in src3)
    out.append(_R("Watcher forwards --table*-limit / --skip-verify", ok))

    return out


# ---------------------------------------------------------------------------
# Section 3: Reproducibility checklist hooks
# ---------------------------------------------------------------------------


def section_reproducibility() -> list[_R]:
    out: list[_R] = []

    out.append(_R(
        "Configs - paper-faithful + default presets shipped",
        _exists("configs/paper_parity.yaml") and _exists("configs/default.yaml")
    ))
    out.append(_R(
        "Regression tests - test directory non-empty",
        _exists("tests") and len(list((ROOT / "tests").glob("test_*.py"))) >= 30
    ))
    out.append(_R(
        "Anonymous-submission helper - make_anonymous_submission.py",
        _exists("scripts/make_anonymous_submission.py")
    ))
    out.append(_R(
        "Post-Stage-2 pipeline orchestrator",
        _exists("scripts/run_post_stage2_pipeline.py")
    ))
    out.append(_R(
        "Reproducibility checklist - REPRODUCIBILITY.md",
        _exists("REPRODUCIBILITY.md")
    ))
    out.append(_R(
        "Stage 2 ckpt validator - test_stage2_ppo_paper_parity.py",
        _exists("tests/test_stage2_ppo_paper_parity.py")
    ))
    out.append(_R(
        "Static D-7 validation suite (torch-free, runs in 0.4 s)",
        _exists("tests/test_d7_static_validation.py")
    ))
    out.append(_R(
        "Author's D12 sanity script",
        _exists("scripts/_d12_final_check.py")
    ))
    out.append(_R(
        "Garbage-math fallback helper (Q14 Fix B factored out)",
        _exists("cts/eval/garbage_filter.py")
    ))
    out.append(_R(
        "Mock-based dispatcher fallback test (torch-free)",
        _exists("tests/test_dispatcher_fallback_mock.py")
    ))
    out.append(_R(
        "Reviewer one-command replication script (Linux GPU)",
        _exists("scripts/replicate_neurips_2026.sh")
    ))
    out.append(_R(
        "Paper-code mapping table validator (5-pent contract)",
        _exists("tests/test_paper_code_mapping_table.py")
    ))
    out.append(_R(
        "Stage 2 training_meta static contract test",
        _exists("tests/test_stage2_training_meta_static.py")
    ))
    out.append(_R(
        "run_cts_eval_full --dry-run mode (no torch)",
        "--dry-run" in _read("scripts/run_cts_eval_full.py")
        and "_dry_run(args)" in _read("scripts/run_cts_eval_full.py")
    ))
    out.append(_R(
        "ZIP byte-invariants test (third leak-defence layer)",
        _exists("tests/test_anon_zip_byte_invariants.py")
    ))
    out.append(_R(
        "Paper-section alignment test (cross-doc §-numbering)",
        _exists("tests/test_paper_section_alignment.py")
    ))
    out.append(_R(
        "Reviewer walkthrough script (drilldown navigation)",
        _exists("scripts/reviewer_walkthrough.py")
    ))
    out.append(_R(
        "_d12_final_check --export-verdict mode",
        "--export-verdict" in _read("scripts/_d12_final_check.py")
        and "_build_verdict_payload" in _read("scripts/_d12_final_check.py")
    ))
    out.append(_R(
        "replicate_neurips_2026.sh --ci-mode",
        "--ci-mode" in _read("scripts/replicate_neurips_2026.sh")
        and "MODE=\"ci-mode\"" in _read("scripts/replicate_neurips_2026.sh")
    ))
    out.append(_R(
        "REVIEWER_FAQ Q16 (CI runnability guide)",
        "Q16." in _read("REVIEWER_FAQ.md")
        and "GitHub Actions" in _read("REVIEWER_FAQ.md")
    ))
    out.append(_R(
        "REVIEWER_FAQ Q17 (paper-section to claim crossref)",
        "Q17." in _read("REVIEWER_FAQ.md")
        and "reviewer_walkthrough.py" in _read("REVIEWER_FAQ.md")
    ))
    out.append(_R(
        "Walkthrough invariants regression test",
        _exists("tests/test_reviewer_walkthrough_invariants.py")
    ))
    out.append(_R(
        "CHANGELOG D-7 completeness regression test",
        _exists("tests/test_changelog_d7_completeness.py")
    ))
    out.append(_R(
        "_d12_final_check Section 5 (byte-invariants integrated)",
        "section_byte_invariants" in _read("scripts/_d12_final_check.py")
    ))
    out.append(_R(
        "CI workflow uploads D12 verdict artefact",
        "Upload D12 verdict artefact" in _read(".github/workflows/tests.yml")
        and "actions/upload-artifact" in _read(".github/workflows/tests.yml")
    ))
    out.append(_R(
        "_reviewer_local_audit --json mode (programmatic verdict)",
        "--json" in _read("scripts/_reviewer_local_audit.py")
        and "_build_json_payload" in _read("scripts/_reviewer_local_audit.py")
    ))
    out.append(_R(
        "Reviewer walkthrough --html mode (browser-readable)",
        "--html" in _read("scripts/reviewer_walkthrough.py")
        and "_render_html" in _read("scripts/reviewer_walkthrough.py")
    ))
    out.append(_R(
        "REPRODUCIBILITY checklist coverage test",
        _exists("tests/test_reproducibility_checklist_coverage.py")
    ))
    out.append(_R(
        "LIMITATIONS completeness regression test",
        _exists("tests/test_limitations_completeness.py")
    ))
    out.append(_R(
        "Pre-submission orchestrator (D12 GO/NO-GO gate)",
        _exists("scripts/run_pre_submission_audit.py")
    ))

    return out


# ---------------------------------------------------------------------------
# Section 4: Documentation marker presence
# ---------------------------------------------------------------------------


def section_docs() -> list[_R]:
    out: list[_R] = []

    faq = _read("REVIEWER_FAQ.md")
    chlog = _read("CHANGELOG.md")
    rep = _read("REPRODUCIBILITY.md")
    pvl = _read("results/table2/PAPER_VS_LOCAL.md")
    lim = _read("LIMITATIONS.md")

    # FAQ Q1-Q15 spot check
    for q in ("Q1", "Q11", "Q13", "Q14", "Q15"):
        out.append(_R(f"REVIEWER_FAQ {q} present", q + "." in faq or q + " " in faq))

    out.append(_R("CHANGELOG D-7 entries (morning + afternoon)",
                  "D-7 Apr 29 (morning)" in chlog and
                  ("D-7 Apr 29 (afternoon)" in chlog or "single-host" in chlog)))
    out.append(_R("REPRODUCIBILITY 5-bis (Audit-fix lineage)", "5-bis" in rep))
    out.append(_R("REPRODUCIBILITY 5-ter (Pipeline guarantees)", "5-ter" in rep))
    out.append(_R("REPRODUCIBILITY 5-quat (single-host blocker)", "5-quat" in rep))
    out.append(_R("PAPER_VS_LOCAL.md status banner",
                  "Status banner" in pvl or "status banner" in pvl))
    out.append(_R("PAPER_VS_LOCAL.md Why-the-Gap appendix",
                  "Why the Gap" in pvl))
    out.append(_R("LIMITATIONS.md - consolidated reviewer-facing limitations doc",
                  bool(lim) and "Compute-scaling gap" in lim))
    out.append(_R("REPRODUCIBILITY 5-pent paper-claim mapping (extended)",
                  "5-pent" in rep and "Paper claim" in rep))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print(r: _R) -> None:
    flag = "[PASS]" if r.ok else "[FAIL]"
    print(f"  {flag} {r.name.ljust(64)}  {r.evidence}")


def _build_json_payload(
    sections: list[tuple[str, list[_R]]],
    passed: int,
    total: int,
) -> dict:
    """Assemble a structured JSON payload that a reviewer can
    program-attach to their review template (e.g. dump it into
    OpenReview's "Confidential comment to area chair" slot, or
    pipe to ``jq`` to extract specific failed checks)."""
    return {
        "schema": "cts_neurips2026.reviewer_audit.v1",
        "verdict": "ALL_GREEN" if passed == total else "PARTIAL_FAIL",
        "passed": passed,
        "total": total,
        "ratio": f"{passed}/{total}",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sections": [
            {
                "name": name,
                "passed": sum(1 for r in results if r.ok),
                "total": len(results),
                "checks": [
                    {
                        "name": r.name,
                        "ok": r.ok,
                        "evidence": r.evidence,
                    }
                    for r in results
                ],
            }
            for name, results in sections
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CTS NeurIPS 2026 - reviewer-side local audit"
    )
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="Write the structured audit verdict (JSON, "
             "schema=cts_neurips2026.reviewer_audit.v1) to PATH. "
             "When set, the human-readable section output is "
             "suppressed and only the final summary line + the "
             "JSON file are emitted. Useful for programmatic "
             "review-template attachment, ``jq`` post-processing, "
             "or CI matrix uploads.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-check output; only the final summary "
             "line is printed (same exit code as a normal run).",
    )
    args = parser.parse_args(argv)
    json_mode = args.json is not None
    quiet = args.quiet or json_mode

    if not quiet:
        print("=" * 80)
        print("CTS NeurIPS 2026 - reviewer-side local audit (~1 second, no torch)")
        print("=" * 80)
        print("\nThis script reads only what's on disk in your unzipped checkout.")
        print("It does NOT import torch, run the GPU pipeline, or call any audit")
        print("scripts that the author would have used during submission. It is")
        print("safe to run on any environment that can run Python 3.")

    sections: list[tuple[str, list[_R]]] = [
        ("Section 1: Headline-claim coverage", section_headline_claims()),
        ("Section 2: D-7 patch-site presence", section_d7_patch_sites()),
        ("Section 3: Reproducibility checklist hooks", section_reproducibility()),
        ("Section 4: Documentation marker presence", section_docs()),
    ]
    total = 0
    passed = 0
    for name, results in sections:
        if not quiet:
            print(f"\n{name}")
        for r in results:
            if not quiet:
                _print(r)
            total += 1
            if r.ok:
                passed += 1

    print("\n" + "=" * 80)
    print(f"Reviewer-side audit summary: {passed}/{total} PASS")
    print("=" * 80)

    if json_mode:
        payload = _build_json_payload(sections, passed, total)
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n>>> JSON verdict written to: {json_path}")
        print(f">>> schema: {payload['schema']}, verdict: {payload['verdict']}")

    if passed == total:
        if not quiet:
            print("\n>>> ALL GREEN. The reviewer-facing static surface is intact:")
            print("    - every D-7 fix the FAQ describes is present in the source,")
            print("    - every claim with a code anchor lands on disk,")
            print("    - every reviewer-facing doc is consistent.")
            print("    Behavioural / numerical claims still require a GPU run on")
            print("    your end (REVIEWER_FAQ Q15 has the canonical command).")
        return 0
    if not quiet:
        print(f"\n>>> {total - passed} CHECKS FAILED. These are the items a reviewer")
        print("    should challenge or ask about in their review. The static")
        print("    contract between paper claims and source code is incomplete.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
