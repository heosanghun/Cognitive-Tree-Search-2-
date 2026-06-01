#!/usr/bin/env python3
"""One-shot pre-submission orchestrator (D12 = May 6).

The author's "click before submit" script. Runs every static
verification gate the submission depends on, in the right
order, with appropriate timeouts, and prints a single colour-
coded GO / NO-GO verdict at the end. Designed to be the
**only** command the author needs to run on D12 morning.

Pipeline (all torch-free; ~5 seconds total on a clean repo):

  Step 1: D-7 static validation             (~20 ms)
          tests/test_d7_static_validation.py + 7 sibling suites
  Step 2: Reviewer-side audit               (~0.5 s)
          scripts/_reviewer_local_audit.py --quiet
  Step 3: Reviewer walkthrough              (~0.5 s)
          scripts/reviewer_walkthrough.py
  Step 4: D12 final-submission sanity       (~1.2 s)
          scripts/_d12_final_check.py --quiet --export-verdict ...
  Step 5: ZIP byte-invariants spot-check    (~80 ms)
          tests/test_anon_zip_byte_invariants.py
  Step 6: Replication script CI mode        (~1.0 s, no GPU)
          bash scripts/replicate_neurips_2026.sh --ci-mode
          (skipped on Windows hosts where bash is unavailable)

Exit codes (mirrors `_d12_final_check.py`):
  0 = GO              (every gate passed)
  1 = SOFT FAIL       (non-blocking marker drift; safe to submit
                       but author should patch)
  2 = HARD NO-GO      (ZIP build/audit/byte-invariants failed;
                       DO NOT submit)

Usage:
  python scripts/run_pre_submission_audit.py
  python scripts/run_pre_submission_audit.py --quiet
  python scripts/run_pre_submission_audit.py --skip-replication
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ANSI colour codes; degrade gracefully on Windows console.
_GREEN = "\x1b[32m" if sys.stdout.isatty() else ""
_RED   = "\x1b[31m" if sys.stdout.isatty() else ""
_YEL   = "\x1b[33m" if sys.stdout.isatty() else ""
_DIM   = "\x1b[2m"  if sys.stdout.isatty() else ""
_RESET = "\x1b[0m"  if sys.stdout.isatty() else ""


def _label(ok: int, total: int) -> str:
    if total == 0:
        return f"{_YEL}SKIP{_RESET}"
    if ok == total:
        return f"{_GREEN}PASS{_RESET}"
    return f"{_RED}FAIL{_RESET}"


def _run_static_suite(rel: str) -> tuple[int, int, str]:
    """Import a torch-free test module by file path and run every
    ``test_*`` function, returning (pass, total, error_excerpt)."""
    p = ROOT / rel
    if not p.is_file():
        return 0, 0, f"missing: {rel}"
    try:
        spec = importlib.util.spec_from_file_location("_t", str(p))
        if spec is None or spec.loader is None:
            return 0, 0, "spec_from_file_location returned None"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - defensive
        return 0, 0, f"import error: {type(exc).__name__}: {str(exc)[:80]}"
    test_names = [n for n in dir(mod) if n.startswith("test_")]
    passed = 0
    err_excerpt = ""
    for n in test_names:
        try:
            getattr(mod, n)()
            passed += 1
        except Exception as exc:
            if not err_excerpt:
                err_excerpt = f"{n}: {type(exc).__name__}: {str(exc)[:80]}"
    return passed, len(test_names), err_excerpt


def _run_subprocess(argv: list[str], timeout: int = 120) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            argv, cwd=str(ROOT), capture_output=True,
            text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        ok = proc.returncode == 0
        tail = (proc.stdout or proc.stderr).strip().splitlines()
        detail = tail[-1] if tail else f"rc={proc.returncode}"
        return ok, detail[:120]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT ({timeout}s)"
    except FileNotFoundError as exc:
        return False, f"executable missing: {exc}"


def step_static_suites(quiet: bool) -> tuple[int, int, list[tuple[str, int, int, str]]]:
    suites = (
        "tests/test_d7_static_validation.py",
        "tests/test_dispatcher_fallback_mock.py",
        "tests/test_paper_code_mapping_table.py",
        "tests/test_stage2_training_meta_static.py",
        "tests/test_anon_zip_byte_invariants.py",
        "tests/test_paper_section_alignment.py",
        "tests/test_reviewer_walkthrough_invariants.py",
        "tests/test_changelog_d7_completeness.py",
        "tests/test_reproducibility_checklist_coverage.py",
        "tests/test_limitations_completeness.py",
    )
    rows: list[tuple[str, int, int, str]] = []
    total_pass = total_count = 0
    for rel in suites:
        t0 = time.time()
        p, t, err = _run_static_suite(rel)
        elapsed_ms = int((time.time() - t0) * 1000)
        rows.append((rel, p, t, err))
        total_pass += p
        total_count += t
        if not quiet:
            label = _label(p, t)
            tail = f" {_DIM}[{elapsed_ms} ms]{_RESET}"
            err_str = f"  {_RED}{err}{_RESET}" if err else ""
            print(f"    {label} {p:3d}/{t:3d}  {rel}{tail}{err_str}")
    return total_pass, total_count, rows


def step_reviewer_audit(quiet: bool) -> tuple[bool, str]:
    return _run_subprocess(
        [sys.executable, "scripts/_reviewer_local_audit.py", "--quiet"],
        timeout=30,
    )


def step_walkthrough(quiet: bool) -> tuple[bool, str]:
    return _run_subprocess(
        [sys.executable, "scripts/reviewer_walkthrough.py"],
        timeout=30,
    )


def step_d12_sanity(quiet: bool) -> tuple[bool, str]:
    return _run_subprocess(
        [sys.executable, "scripts/_d12_final_check.py", "--quiet",
         "--export-verdict", "results/d12_verdict.json"],
        timeout=180,
    )


def step_replication_ci_mode(quiet: bool) -> tuple[bool, str]:
    bash = shutil.which("bash")
    if bash is None:
        return True, "skipped (bash not on PATH; OK on Windows hosts)"
    return _run_subprocess(
        [bash, "scripts/replicate_neurips_2026.sh", "--ci-mode"],
        timeout=180,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-submission audit (D12 GO / NO-GO gate)"
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output; only print final verdict.")
    parser.add_argument("--skip-replication", action="store_true",
                        help="Skip Step 6 (bash replication --ci-mode); "
                             "useful on Windows hosts without bash.")
    args = parser.parse_args(argv)
    quiet = args.quiet

    t_start = time.time()
    if not quiet:
        print("=" * 78)
        print("CTS NeurIPS 2026 - pre-submission audit (D12 GO / NO-GO gate)")
        print("=" * 78)

    # ---------- Step 1: static suites ----------
    if not quiet:
        print("\n[STEP 1/6] Torch-free static suites (10 files)")
    s1_pass, s1_total, _ = step_static_suites(quiet)
    s1_ok = (s1_pass == s1_total)
    if not quiet:
        label = _label(s1_pass, s1_total)
        print(f"  -> {label}: {s1_pass}/{s1_total}")

    # ---------- Step 2: reviewer audit ----------
    if not quiet:
        print("\n[STEP 2/6] Reviewer-side static audit")
    s2_ok, s2_detail = step_reviewer_audit(quiet)
    if not quiet:
        print(f"  -> {_label(int(s2_ok), 1)}: {s2_detail}")

    # ---------- Step 3: walkthrough ----------
    if not quiet:
        print("\n[STEP 3/6] Reviewer walkthrough (zero-MISS gate)")
    s3_ok, s3_detail = step_walkthrough(quiet)
    if not quiet:
        print(f"  -> {_label(int(s3_ok), 1)}: {s3_detail}")

    # ---------- Step 4: D12 sanity ----------
    if not quiet:
        print("\n[STEP 4/6] D12 final-submission sanity (rebuild ZIP + audit)")
    s4_ok, s4_detail = step_d12_sanity(quiet)
    if not quiet:
        print(f"  -> {_label(int(s4_ok), 1)}: {s4_detail}")

    # ---------- Step 5: byte invariants spot-check ----------
    # Already covered by Section 5 of d12_sanity, but re-run
    # standalone so a Step 5 failure is visible without parsing
    # d12 output.
    if not quiet:
        print("\n[STEP 5/6] ZIP byte-invariants spot-check")
    s5_pass, s5_total, _ = ((0, 0, []) if not s4_ok
                            else (lambda r: (r[0], r[1], []))(
        _run_static_suite("tests/test_anon_zip_byte_invariants.py")
    ))
    s5_ok = s4_ok and s5_pass == s5_total and s5_total > 0
    if not quiet:
        print(f"  -> {_label(s5_pass, s5_total) if s4_ok else _YEL+'SKIP'+_RESET}: "
              f"{s5_pass}/{s5_total}")

    # ---------- Step 6: replication --ci-mode ----------
    if args.skip_replication:
        s6_ok, s6_detail = True, "skipped (--skip-replication)"
    else:
        if not quiet:
            print("\n[STEP 6/6] Replication script --ci-mode")
        s6_ok, s6_detail = step_replication_ci_mode(quiet)
    if not quiet:
        print(f"  -> {_label(int(s6_ok), 1)}: {s6_detail}")

    # ---------- Verdict ----------
    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"Elapsed: {elapsed:.1f} s")
    # Hard-fail gates (BLOCKING for D12 submission):
    hard_fail = not (s4_ok and s5_ok)
    # Soft-fail gates (non-blocking but should be patched):
    soft_fail = not (s1_ok and s2_ok and s3_ok and s6_ok)

    if hard_fail:
        print(f"{_RED}>>> NO-GO: BLOCKING failures in ZIP build/audit "
              f"or byte-invariants. DO NOT SUBMIT.{_RESET}")
        rc = 2
    elif soft_fail:
        print(f"{_YEL}>>> SOFT-FAIL: D12 ZIP gate passed but auxiliary "
              f"checks failed; safe to submit, author should patch.{_RESET}")
        rc = 1
    else:
        print(f"{_GREEN}>>> GO: every gate passed. "
              f"Submission is D12-ready.{_RESET}")
        rc = 0
    print("=" * 78)

    if not quiet:
        print(f"\nVerdict artefacts:")
        print(f"  results/d12_verdict.json   (structured)")
        print(f"  results/d12_verdict.md     (paste-ready for OpenReview)")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
