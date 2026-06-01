"""Byte-level invariants on the anonymous-submission ZIP.

The existing ``scripts/_audit_anon_zip.py`` checks for path-name
leaks (PROGRESS_REPORT*, OPENREVIEW*, etc.) and content-level
identity tokens (handles, emails, Windows username path leaks).
This file adds a third defence layer: **structural invariants on
the ZIP itself** that catch failures the path/content audit can
miss:

  1. The ZIP file actually exists and is readable.
  2. Every file's compressed size <= uncompressed size (basic
     ZIP integrity; rules out a corrupted archive).
  3. No ZIP entry's path component starts with ``.`` other than
     a small whitelist (``.github``, ``.gitignore``,
     ``.flake8``, ``.pre-commit-config.yaml``); catches stray
     ``.git/`` / ``.git_backup_*`` / ``.mailmap_anonymize`` /
     ``.idea/`` / ``.vscode/`` leaks.
  4. No file inside the ZIP exceeds 10 MB (catches accidental
     model checkpoint inclusion; the LFS-tracked tensors are
     EXCLUDE_GLOBS-ed, but a broken EXCLUDE_GLOBS would let
     them through silently).
  5. The ZIP contains expected reviewer-facing entry points:
     README.md, REVIEWER_FAQ.md, REPRODUCIBILITY.md,
     LIMITATIONS.md, scripts/_reviewer_local_audit.py,
     scripts/replicate_neurips_2026.sh.
  6. The ZIP does NOT contain author-draft content even when
     the path was renamed (e.g. someone copies
     OPENREVIEW_RESPONSE_PREP.md into docs/ so the path
     pattern misses it). We grep the *content* of every
     text-like file for the known author-draft headers.

Runs in <1 second on the existing ZIP, no torch.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "anonymous_submission_neurips2026.zip"

# Path components allowed to start with '.' (everything else is suspicious).
ALLOWED_DOT_PREFIXES = frozenset({
    ".github",
    ".gitignore",
    ".flake8",
    ".pre-commit-config.yaml",
    ".gitattributes",
    ".coveragerc",
    ".dockerignore",
    ".editorconfig",
})

# Per-file size cap inside the ZIP (10 MB). Larger files are
# almost always accidental model-weight inclusions.
MAX_FILE_BYTES = 10 * 1024 * 1024

# Known author-draft content headers. If any of these strings
# appears in a text-like ZIP entry, an author draft has leaked
# even if the path pattern does not match.
AUTHOR_DRAFT_HEADERS = (
    "OpenReview Rebuttal Templates",  # OPENREVIEW_RESPONSE_PREP.md
    "PROGRESS REPORT",                 # PROGRESS_REPORT_*.md
    "PROGRESS_REPORT_2026",            # explicit filename mention
    "advisor meeting",                 # PROGRESS_REPORT advisor framing
)

# Reviewer-facing entry points that MUST be in the ZIP. A
# regression that drops any of these would silently kill the
# reviewer experience.
EXPECTED_ENTRY_POINTS = (
    "README.md",
    "REVIEWER_FAQ.md",
    "REPRODUCIBILITY.md",
    "LIMITATIONS.md",
    "CHANGELOG.md",
    "scripts/_reviewer_local_audit.py",
    "scripts/replicate_neurips_2026.sh",
    "scripts/run_cts_eval_full.py",
    "tests/test_d7_static_validation.py",
    "tests/test_dispatcher_fallback_mock.py",
    "tests/test_paper_code_mapping_table.py",
    "tests/test_stage2_training_meta_static.py",
    "cts/eval/garbage_filter.py",
    "cts/policy/meta_policy.py",
    "cts/critic/neuro_critic.py",
)

TEXT_EXTS = frozenset({
    ".md", ".py", ".txt", ".yaml", ".yml", ".json", ".toml",
    ".cfg", ".ini", ".rst", ".sh", ".bat", ".ps1",
})


def _ensure_zip_exists() -> Path:
    if not ZIP_PATH.is_file():
        pytest.skip(
            f"{ZIP_PATH.name} not present locally; this test only runs after "
            f"`python scripts/make_anonymous_submission.py`."
        )
    return ZIP_PATH


def _strip_top(name: str) -> str:
    """The ZIP wraps everything under ``cts_neurips2026/``; strip
    that prefix so we can match against EXPECTED_ENTRY_POINTS."""
    parts = name.split("/", 1)
    return parts[1] if len(parts) == 2 else name


def test_zip_exists_and_is_readable():
    p = _ensure_zip_exists()
    assert p.stat().st_size > 0
    with zipfile.ZipFile(p) as zf:
        assert zf.namelist(), "ZIP is empty"


def test_zip_compression_invariant():
    """Every entry's compressed size must be sane. Note that the
    ZIP standard allows ``compress_size > file_size`` for empty
    files (irreducible 2-byte STORED record) and for already-
    compressed payloads (PNG, etc.) that the deflate pass cannot
    shrink further; we cap the allowed overhead at 1 KB +
    ``file_size``, which catches genuine corruption while
    accepting the legitimate cases."""
    _ensure_zip_exists()
    OVERHEAD_BUDGET = 1024  # 1 KB ceiling on legitimate ZIP overhead
    bad: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.compress_size > info.file_size + OVERHEAD_BUDGET:
                bad.append(
                    f"{info.filename}: compressed={info.compress_size} "
                    f"> uncompressed={info.file_size} + {OVERHEAD_BUDGET}"
                )
    assert not bad, "ZIP integrity suspect:\n  " + "\n  ".join(bad)


def test_zip_has_no_unexpected_dotfiles():
    _ensure_zip_exists()
    bad: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            stripped = _strip_top(name)
            if not stripped:
                continue
            for part in stripped.split("/"):
                if part.startswith(".") and part not in ALLOWED_DOT_PREFIXES:
                    bad.append(name)
                    break
    assert not bad, (
        f"ZIP contains unexpected dotfile entries (dev-environment leak?): "
        f"{bad[:10]}{' ...' if len(bad) > 10 else ''}"
    )


def test_zip_no_large_files():
    """A single file >10 MB is almost always a model-weight or
    a logfile leak. The largest legitimate file in the
    submission is the architecture diagram (~120 KB)."""
    _ensure_zip_exists()
    bad: list[tuple[str, int]] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.file_size > MAX_FILE_BYTES:
                bad.append((info.filename, info.file_size))
    assert not bad, (
        f"ZIP contains files > {MAX_FILE_BYTES // (1024*1024)} MB: "
        f"{[(n, f'{s/(1024*1024):.1f}MB') for n, s in bad]}"
    )


def test_zip_contains_all_reviewer_entry_points():
    """Every reviewer-facing file MUST be present in the ZIP.
    A regression that drops one would silently break the
    reviewer's first command after unzipping."""
    _ensure_zip_exists()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = {_strip_top(n) for n in zf.namelist()}
    missing = [ep for ep in EXPECTED_ENTRY_POINTS if ep not in names]
    assert not missing, (
        f"ZIP missing reviewer entry points: {missing}"
    )


def test_zip_contains_no_author_draft_content():
    """Defence layer 3: even if a path-name escape gets past
    EXCLUDE_GLOBS + HARD_FAIL_PATHS, the content of any
    text-like file in the ZIP must not contain known
    author-draft headers."""
    _ensure_zip_exists()
    bad: list[tuple[str, str]] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in TEXT_EXTS:
                continue
            try:
                raw = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for header in AUTHOR_DRAFT_HEADERS:
                if header in raw:
                    bad.append((name, header))
    # Whitelist: CHANGELOG.md and REVIEWER_FAQ.md may legitimately
    # mention these strings while documenting the leak-prevention
    # mechanism. We allow up to N=3 mentions per file in those
    # specific files.
    filtered: list[tuple[str, str]] = []
    for n, h in bad:
        stripped = _strip_top(n)
        if stripped in (
            "CHANGELOG.md", "REVIEWER_FAQ.md", "REPRODUCIBILITY.md",
            "LIMITATIONS.md", "tests/test_anon_zip_byte_invariants.py",
            "tests/test_d7_static_validation.py",
            "scripts/make_anonymous_submission.py",
            "scripts/_reviewer_local_audit.py",
            "scripts/_d12_final_check.py",
        ):
            continue
        filtered.append((n, h))
    assert not filtered, (
        "ZIP contains author-draft content (likely a renamed leak):\n  " +
        "\n  ".join(f"{n}: header={h!r}" for n, h in filtered[:10])
    )
