#!/usr/bin/env python3
"""Build an anonymous, NeurIPS-double-blind-safe ZIP of this repository.

Produces ``anonymous_submission_neurips2026.zip`` in the repo root, suitable for:
  - upload to https://anonymous.4open.science (file mode), OR
  - direct attachment as supplementary material to the OpenReview submission.

The zip is fully self-contained for code review (no large weights, no datasets,
no git history, no PDF, no caches). Reviewers can recreate weights/data via
``scripts/run_stage1_openmath.py``, ``scripts/run_stage2_math_ppo.py``, and
``scripts/download_all_benchmarks.py`` exactly as documented in README.md.

Excluded paths (with rationale):
  - .git/                   author identity in commit history
  - artifacts/              16 GB+ trained checkpoints (regenerable)
  - .hf_cache/, gemma-*/    15 GB+ model weights (Hugging Face)
  - data/                   downloadable via download scripts
  - doc/                    contains paper PDF with author metadata
  - results/                run-specific outputs (regenerable)
  - terminals/, .vscode/, .cursor/   IDE/session artifacts
  - __pycache__/, *.pyc, *.pyo       Python bytecode
  - *.pt, *.bin, *.safetensors        binary weights
  - cursor_*.md, *.canvas.tsx        development scratch files
"""

from __future__ import annotations

import fnmatch
import sys
import zipfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ZIP = REPO_ROOT / "anonymous_submission_neurips2026.zip"

# Top-level directories/files to EXCLUDE entirely. ``results`` is allowed
# *partially* (see RESULTS_INCLUDE_GLOBS) so that the small reviewer-facing
# JSON / Markdown artifacts (Table-2 reproduction summaries) ride along
# without bloating the zip with raw model outputs.
EXCLUDE_TOP = {
    ".git", ".github", ".vscode", ".cursor", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".ipynb_checkpoints",
    # Author-side agent rules (SSH/GPU-server allocation policy). Not part
    # of the reviewer-facing artifact, and its content references the
    # authors' local infrastructure — must never ship in the anonymous ZIP
    # (also enforced by tests/test_anon_zip_byte_invariants.py's
    # no-unexpected-dotfiles invariant).
    ".agents",
    "artifacts", ".hf_cache", "gemma-4-E4B", "gemma-4-E4B-it",
    "data", "doc", "terminals",
    # Author-facing run logs may contain Windows redirects, GPU device
    # fingerprints (torch/cu118 versions, "W0426 ... Windows" stderr),
    # local timezone-stamped filenames, and potentially partial paths
    # that a determined reviewer could correlate. Reviewers regenerate
    # logs locally when they re-run training; the directory itself is
    # not part of the reproducible artifact.
    "logs",
    "__pycache__", "node_modules", "venv", ".venv", "env",
    ".git_backup_pre_anonymize",  # never ship the pre-anonymize backup
    # If a previously-built ZIP is unzipped alongside the working tree
    # (e.g. for review/audit/IDE inspection), do not recursively re-include
    # it inside a fresh build.
    "anonymous_submission_neurips2026",
}

# Glob patterns to EXCLUDE at any depth.
EXCLUDE_GLOBS = [
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    "*.pt", "*.bin", "*.safetensors", "*.ckpt", "*.pth",
    "*.pdf", "*.zip", "*.tar", "*.tar.gz", "*.tgz",
    "*.bak", "*.bak_*",  # backup files (e.g. anonymous_submission*.zip.bak_pre_*)
    ".DS_Store", "Thumbs.db",
    "cursor_*.md", "*.canvas.tsx",
    "anonymous_submission_neurips2026.zip",
    ".mailmap_anonymize",
    "_filter_repo_callback.py",
    # Internal developer tooling that itself contains the identity-leak
    # patterns it is designed to detect; not useful to reviewers.
    "_audit_anon_zip.py",
    # Author-facing maintainer tool for continuous GitHub sync; mentions
    # the upstream git remote and is not part of the reproduction surface.
    "sync_to_github.py",
    # Author-facing planning / triage documents.  These intentionally cite
    # the personal GitHub handle, local paths, and unresolved issue lists.
    # Reviewers should never see them; the public-facing material lives in
    # README.md, REPRODUCIBILITY.md, REVIEWER_FAQ.md, and CHANGELOG.md.
    "SUBMISSION_GUIDE_D12.md",
    "NEXT_TASKS_PRIORITIZED.md",
    "PAPER_VS_LOCAL_FINAL.md",
    "PAPER_VS_LOCAL_INTUITIVE.md",
    "PAPER_CONSISTENCY_AUDIT.md",
    "ROOT_CAUSE_ANALYSIS.md",
    "EXPERIMENTAL_RESULTS.md",
    "EXPERIMENTS.md",
    # Author-facing progress reports (advisor-meeting prep, contains
    # honest scoring of NeurIPS readiness in author's working language).
    # Reviewer-facing equivalents live in REVIEWER_FAQ.md and
    # results/table2/PAPER_VS_LOCAL.md.
    "PROGRESS_REPORT_*.md",
    # Author-side rebuttal templates (D-7 Apr 29 evening): contain
    # candid internal language and must never appear in the
    # double-blind ZIP. Reviewer-facing limitations live in
    # LIMITATIONS.md and REVIEWER_FAQ.md.
    "OPENREVIEW_RESPONSE_PREP.md",
    "NEXT_TASKS_*.md",
    "PAPER_VS_LOCAL_FINAL.md",
    "PAPER_VS_LOCAL_INTUITIVE.md",
    "PAPER_CONSISTENCY_AUDIT.md",
    "ROOT_CAUSE_ANALYSIS.md",
    "EXPERIMENTAL_RESULTS.md",
    # Historical author-side pipeline manifest (records absolute local
    # working-directory paths and now-removed cloud summary phases);
    # not reviewer-relevant.
    "manifest.json",
    # Author-side run snapshots that contain absolute local paths,
    # PARTIAL_FAIL pipeline_status entries, and other host-specific
    # artefacts that are not reviewer-relevant. Mirrors .gitignore.
    "post_stage2_*",
    "local_gemma4_partial*",
    "local_gemma4_plan_i",
    "preflight_*",
    "smoke_preflight",
    "p0_quick_*",
    "table2_re*",
]

# Whitelist patterns under ``results/`` — only these reviewer-relevant
# artifacts are included; everything else under ``results/`` is dropped.
RESULTS_INCLUDE_GLOBS = [
    "table2_results.json",
    "PAPER_VS_LOCAL.md",
    "*.md",
    "*.json",
    # Small sweep manifests (e.g. sweep_K_plan.txt, sweep_W_plan.txt)
    # are reviewer-relevant: they declare exactly which (K, seed,
    # benchmark) jobs the sweep would launch, which is part of the
    # paper's ablation reproducibility surface.
    "*.txt",
]
# Cap each kept file under ``results/`` to keep the zip small.
RESULTS_FILE_MAX_BYTES = 256 * 1024  # 256 KB


def _path_matches_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def is_excluded(path: Path) -> bool:
    """Return True if `path` (relative to repo root) should be excluded.

    ``results/`` is treated specially: only small reviewer-relevant files
    matching :data:`RESULTS_INCLUDE_GLOBS` are kept, and each kept file
    must be smaller than :data:`RESULTS_FILE_MAX_BYTES`. The size check
    is performed by :func:`iter_files` because :func:`is_excluded` does
    not have access to file size at call time.

    Defense-in-depth: any subdirectory under ``results/`` that begins with
    an underscore is treated as author-internal (debug dumps, smoke-test
    scratchpads) and is excluded wholesale, regardless of file extension.
    Such directories often contain Python tracebacks with local OS paths
    (e.g. ``C:\\Users\\<name>\\AppData\\Local\\Temp\\...``) that would
    de-anonymise the submission if they leaked through the
    ``RESULTS_INCLUDE_GLOBS`` whitelist.
    """
    parts = path.parts
    if parts and parts[0] in EXCLUDE_TOP:
        return True
    if parts and parts[0] == "results":
        if len(parts) >= 2 and parts[1].startswith("_"):
            return True
        if not _path_matches_any(parts[-1], RESULTS_INCLUDE_GLOBS):
            return True
    for part in parts:
        for pat in EXCLUDE_GLOBS:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if is_excluded(rel):
            continue
        # Per-file size cap for ``results/`` whitelist (keeps zip small).
        if rel.parts and rel.parts[0] == "results":
            try:
                if p.stat().st_size > RESULTS_FILE_MAX_BYTES:
                    continue
            except OSError:
                continue
        yield p


def main() -> int:
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    files = list(iter_files(REPO_ROOT))
    total_bytes = sum(f.stat().st_size for f in files)

    print(f"[anon-zip] repo root : {REPO_ROOT}")
    print(f"[anon-zip] output    : {OUT_ZIP.name}")
    print(f"[anon-zip] files     : {len(files)}")
    print(f"[anon-zip] payload   : {total_bytes / 1024 / 1024:.1f} MB (uncompressed)")
    print()

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for fp in files:
            rel = fp.relative_to(REPO_ROOT)
            zf.write(fp, arcname=str(Path("cts_neurips2026") / rel))

    out_size = OUT_ZIP.stat().st_size
    print(f"[anon-zip] DONE      : {OUT_ZIP}")
    print(f"[anon-zip] zip size  : {out_size / 1024 / 1024:.1f} MB (compressed)")
    print()
    print("Next steps for reviewers:")
    print("  1. Upload the zip to https://anonymous.4open.science (file upload mode)")
    print("     OR attach directly to the OpenReview supplementary material slot.")
    print("  2. Reviewers reproduce weights/data via:")
    print("       python scripts/download_all_benchmarks.py")
    print("       python scripts/run_stage1_openmath.py --lora --device cuda:0")
    print("       python scripts/run_stage2_math_ppo.py --stage1-ckpt artifacts/stage1_last.pt --device cuda:0")
    print("  3. End-to-end re-experiment (single 24 GB GPU, ~4 h):")
    print("       see README.md > Local Reproduction Snapshot > Re-experiment #1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
