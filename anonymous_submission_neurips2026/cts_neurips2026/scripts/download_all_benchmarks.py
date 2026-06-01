#!/usr/bin/env python3
"""Download all 5 benchmark datasets for Table 2 reproduction."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def download_humaneval():
    """Download HumanEval from HuggingFace."""
    out = DATA / "humaneval"
    out.mkdir(exist_ok=True)
    target = out / "test.jsonl"
    if target.exists():
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  HumanEval already exists: {n} problems")
        return

    print("  Downloading HumanEval...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        with open(target, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"  Saved {len(ds)} problems -> {target}")
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Creating from openai/human-eval GitHub fallback...")
        _humaneval_fallback(target)


def _humaneval_fallback(target: Path):
    """Fallback: download raw HumanEval from GitHub."""
    import urllib.request
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    gz_path = target.parent / "HumanEval.jsonl.gz"
    try:
        urllib.request.urlretrieve(url, gz_path)
        import gzip
        with gzip.open(gz_path, "rt", encoding="utf-8") as gz, open(target, "w", encoding="utf-8") as f:
            for line in gz:
                f.write(line)
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  Downloaded {n} problems from GitHub -> {target}")
    except Exception as e2:
        print(f"  GitHub fallback also failed: {e2}")


def download_aime(target_year: int = 2026):
    """Download AIME problems for ``target_year`` (paper §7.1: 'AIME 2026').

    The default Hugging Face source ``AI-MO/aimo-validation-aime`` bundles
    AIME 2022 + 2023 + 2024. For the paper's headline benchmark (AIME 2026)
    those URLs do not exist on AoPS Wiki at scrape time, so this loader
    relies on the manually-collected jsonl at
    ``data/aime/test_2026.jsonl`` (combined I + II = 30 problems) and
    refuses to silently substitute earlier years.

    Strict-year mode: if fewer than 30 ``target_year`` problems are
    available, raise ``RuntimeError``. The earlier silent-fallback to
    AIME 2024 was the cause of a critical paper-vs-local data mismatch
    and is now disabled.
    """
    out = DATA / "aime"
    out.mkdir(exist_ok=True)
    target = out / "test.jsonl"
    target_year_path = out / f"test_{target_year}.jsonl"

    if target.exists() and target_year_path.exists():
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  AIME already exists: {n} problems  ({target})")
        if n < 30:
            print(
                f"  WARN: AIME {target_year} has only {n} problems on disk; "
                f"paper expects 30 (15 I + 15 II)."
            )
        return

    print(f"  Loading AIME {target_year} from manually-collected source...")
    if not target_year_path.exists():
        raise RuntimeError(
            f"AIME {target_year} source file not found at {target_year_path}. "
            f"Please run the AoPS Wiki collection step (see "
            f"`data/aime/README.md`) before invoking this loader. "
            f"Silent fallback to earlier years is disabled to keep "
            f"paper §7.1 alignment honest."
        )
    rows = []
    for line in open(target_year_path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if len(rows) < 30:
        raise RuntimeError(
            f"AIME {target_year} only has {len(rows)} problems collected; "
            f"paper expects 30 (15 I + 15 II). Refusing to write a partial "
            f"benchmark to {target} so that no run can silently report "
            f"under-sampled AIME numbers."
        )
    with open(target, "w", encoding="utf-8") as f:
        for it in rows:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  Saved {len(rows)} AIME {target_year} problems -> {target}")


_AOPS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _aops_fetch(url: str, timeout: float = 25.0) -> str:
    """Fetch an AoPS Wiki page with a browser User-Agent.

    AoPS returns 403 to default urllib UA, so we always set a real-looking
    User-Agent. Returns the HTML body decoded as UTF-8 (errors='ignore').
    """
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": _AOPS_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def _aops_extract_problems(html: str):
    """Extract the 15 problem texts from an AoPS '<year>_AIME_<I|II>_Problems' page.

    Strategy: locate every ``id="Problem_N"`` heading, take the slice up to the
    next ``id="Problem_M"`` (or ``id="See_also"`` / end-of-body), then convert:

    * ``<img ... alt="$LATEX$" ...>``  -> ``$LATEX$``  (LaTeX from the alt attr)
    * ``<a ... title="...Problem_N">Solution</a>`` block -> dropped
    * Remaining HTML tags  -> stripped
    * HTML entities        -> unescaped

    Returns a list of (problem_number, problem_text) sorted by problem_number.
    """
    import html as _html
    import re as _re

    starts = list(_re.finditer(r'id="Problem_(\d+)"', html))
    if not starts:
        return []
    end_marker = _re.search(r'id="See_also"|<div class="printfooter">', html)
    end_pos = end_marker.start() if end_marker else len(html)

    problems = []
    for i, m in enumerate(starts):
        num = int(m.group(1))
        chunk_start = m.end()
        chunk_end = starts[i + 1].start() if i + 1 < len(starts) else end_pos
        chunk = html[chunk_start:chunk_end]
        # Cut off at the next ``<h2>`` heading (the chunk_end above stops
        # *inside* the next ``<span class="mw-headline" id="Problem_M">``,
        # so leftover ``<h2><span class="mw-headline" `` text would otherwise
        # leak into the problem body).
        next_h2 = chunk.find("<h2")
        if next_h2 != -1:
            chunk = chunk[:next_h2]
        # Strip the trailing "Solution" link block if present. AoPS uses
        # ``<a href="...Problem_N">Solution</a>`` (sometimes with a leading
        # space inside the anchor) so we match leniently.
        sol_link = _re.search(
            r'<p>\s*<a [^>]*?>\s*Solution\s*</a>\s*</p>',
            chunk,
            flags=_re.IGNORECASE,
        )
        if sol_link:
            chunk = chunk[: sol_link.start()]
        # Replace LaTeX images with their alt text. AoPS uses both
        # ``alt="$...$"`` (inline ``latex``) and ``alt="\[...\]"`` (display
        # ``latexcenter``); we capture either by grabbing the entire alt
        # attribute and emitting a `` $...$ `` wrapper so downstream tokenizers
        # see uniform inline math.
        def _img_to_latex(mm):
            alt = mm.group(1)
            stripped = alt.strip()
            if stripped.startswith("$") and stripped.endswith("$"):
                return " " + stripped + " "
            if stripped.startswith("\\[") and stripped.endswith("\\]"):
                return " $" + stripped[2:-2].strip() + "$ "
            if stripped.startswith("\\(") and stripped.endswith("\\)"):
                return " $" + stripped[2:-2].strip() + "$ "
            return " " + stripped + " "
        chunk = _re.sub(
            r'<img [^>]*?alt="([^"]*)"[^>]*?>',
            _img_to_latex,
            chunk,
        )
        # Strip all remaining HTML tags
        chunk = _re.sub(r'<[^>]+>', " ", chunk)
        chunk = _html.unescape(chunk)
        # Drop the "Problem N" heading prefix the chunk inherits from the
        # ``<span class="mw-headline" id="Problem_N">Problem N</span>`` block.
        chunk = _re.sub(r"^\s*>?\s*Problem\s+\d+\s*", "", chunk)
        # Drop trailing "Solution" word artifact and collapse whitespace
        chunk = _re.sub(r"\s+", " ", chunk).strip()
        chunk = _re.sub(r"\s*Solution\s*$", "", chunk).strip()
        if chunk:
            problems.append((num, chunk))
    return sorted(problems, key=lambda t: t[0])


def _aops_extract_answer_key(html: str):
    """Extract 15 zero-padded 3-digit answer strings from an AoPS Answer Key page.

    The answer key body is a single ``<ol>`` whose ``<li>`` items are the
    integer answers in problem order. AIME answers are always 0-999, so we
    zero-pad to 3 digits to match the existing ``test_2026.jsonl`` format.
    """
    import re as _re

    m = _re.search(r"<ol>(.*?)</ol>", html, flags=_re.DOTALL)
    if not m:
        return []
    # Some answer keys annotate dual-accepted answers ("080 or 081 (both
    # were accepted)") -- take the FIRST integer in each <li> block so the
    # parser stays robust across all 10 (year, exam) pairs.
    items = _re.findall(r"<li>(.*?)</li>", m.group(1), flags=_re.DOTALL)
    out = []
    for it in items:
        first_int = _re.search(r"\d+", it)
        if first_int is not None:
            out.append(f"{int(first_int.group(0)):03d}")
    return out


def _build_aime_train_row(year: int, exam: str, num: int, text: str, answer: str) -> dict:
    """Construct a JSONL row matching the schema requested in the screening task."""
    label = "I" if exam == "I" else "II"
    return {
        "id": f"aime_{year}_{label}_{num}",
        "problem": text,
        "answer": answer,
        "year": year,
        "exam": label,
        "source": "aops_wiki",
        "url": (
            f"https://artofproblemsolving.com/wiki/index.php?title="
            f"{year}_AIME_{label}_Problems/Problem_{num}"
        ),
    }


def _placeholder_aime_train(years, exams):
    """Deterministic non-math placeholders so downstream code can run offline.

    These rows are CLEARLY marked as placeholders (id prefix
    ``placeholder_``, source ``placeholder``) so the contamination screen
    and any downstream training loop can refuse to use them. They are
    NOT fake math problems -- they are short distinguishable strings
    that will trivially fail any math evaluator.
    """
    rows = []
    for year in years:
        for exam in exams:
            for n in range(1, 16):
                rows.append({
                    "id": f"placeholder_aime_{year}_{exam}_{n}",
                    "problem": (
                        f"PLACEHOLDER AIME {year} {exam} Problem {n}: "
                        f"network access to artofproblemsolving.com was "
                        f"unavailable when this file was generated. "
                        f"Re-run scripts/download_all_benchmarks.py with "
                        f"network access to populate the real text."
                    ),
                    "answer": "000",
                    "year": year,
                    "exam": exam,
                    "source": "placeholder",
                    "url": "",
                })
    return rows


def download_aime_train_2019_2023():
    """Build ``data/aime/train_2019_2023.jsonl`` from AoPS Wiki.

    The paper §7.1 holds out AIME 2024/2025/2026 for evaluation, so the
    Stage 2 PPO training pool is restricted to AIME 2019-2023 (5 years
    x 2 exams x 15 problems = 150 items). Source is the same AoPS Wiki
    used for ``test_2026.jsonl``; if the network is unreachable we fall
    back to a clearly-marked placeholder JSONL so the contamination
    screen can still be exercised on disk.

    The function is idempotent: if the target already exists with at
    least 150 non-placeholder rows it returns immediately.
    """
    out = DATA / "aime"
    out.mkdir(exist_ok=True)
    target = out / "train_2019_2023.jsonl"

    if target.exists():
        rows = [json.loads(l) for l in open(target, "r", encoding="utf-8") if l.strip()]
        real = [r for r in rows if r.get("source") != "placeholder"]
        if len(real) >= 150:
            print(
                f"  AIME train 2019-2023 already exists: {len(real)} real "
                f"rows (+{len(rows) - len(real)} placeholders) -> {target}"
            )
            return
        print(
            f"  AIME train 2019-2023 exists but only has {len(real)} real "
            f"rows; refetching."
        )

    years = [2019, 2020, 2021, 2022, 2023]
    exams = ["I", "II"]
    fetched = []
    failures = []
    for year in years:
        for exam in exams:
            problems_url = (
                f"https://artofproblemsolving.com/wiki/index.php?title="
                f"{year}_AIME_{exam}_Problems"
            )
            answers_url = (
                f"https://artofproblemsolving.com/wiki/index.php?title="
                f"{year}_AIME_{exam}_Answer_Key"
            )
            try:
                p_html = _aops_fetch(problems_url)
                a_html = _aops_fetch(answers_url)
                problems = _aops_extract_problems(p_html)
                answers = _aops_extract_answer_key(a_html)
                if len(problems) < 15 or len(answers) < 15:
                    raise RuntimeError(
                        f"parsed {len(problems)} problems / {len(answers)} "
                        f"answers for {year} {exam} (expected 15/15)"
                    )
                for (num, text), ans in zip(problems[:15], answers[:15]):
                    fetched.append(_build_aime_train_row(year, exam, num, text, ans))
                print(f"  fetched {year} AIME {exam}: 15 problems")
            except Exception as e:
                failures.append((year, exam, str(e)))
                print(f"  WARN: {year} AIME {exam} failed: {e}")

    if len(fetched) >= 150:
        with open(target, "w", encoding="utf-8") as f:
            for row in fetched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved {len(fetched)} real AIME 2019-2023 problems -> {target}")
        return

    print(
        f"  ERROR: only fetched {len(fetched)}/150 problems; writing a "
        f"PLACEHOLDER train set so downstream code can still run. "
        f"Re-run with network access to populate real data."
    )
    placeholders = _placeholder_aime_train(years, exams)
    out_rows = fetched + [
        p for p in placeholders
        if not any(
            f["year"] == p["year"] and f["exam"] == p["exam"]
            and f["id"].endswith("_" + p["id"].split("_")[-1])
            for f in fetched
        )
    ]
    with open(target, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"  Wrote {len(out_rows)} rows ({len(fetched)} real, "
        f"{len(out_rows) - len(fetched)} placeholders) -> {target}"
    )
    if failures:
        print("  failures:")
        for y, e, msg in failures:
            print(f"    {y} {e}: {msg}")


def download_aime_eval_2024_2025():
    """Build ``data/aime/test_2024_2025.jsonl`` from AoPS Wiki.

    Paper §7.4 'Extended AIME validation' uses AIME 2024 + 2025 + 2026
    (90 problems = 3 years × 2 exams × 15) for Table 17. AIME 2026 is
    already collected in ``data/aime/test_2026.jsonl``. This function
    collects the missing 60 items (2024 + 2025) into a separate JSONL so
    downstream Table 17 reproduction can union them with test_2026.

    Same idempotency / placeholder semantics as
    ``download_aime_train_2019_2023``: re-runs are no-ops when the target
    already has 60+ real rows; network failures emit clearly-marked
    placeholders rather than partial real data.
    """
    out = DATA / "aime"
    out.mkdir(exist_ok=True)
    target = out / "test_2024_2025.jsonl"

    if target.exists():
        rows = [json.loads(l) for l in open(target, "r", encoding="utf-8") if l.strip()]
        real = [r for r in rows if r.get("source") != "placeholder"]
        if len(real) >= 60:
            print(
                f"  AIME eval 2024-2025 already exists: {len(real)} real "
                f"rows (+{len(rows) - len(real)} placeholders) -> {target}"
            )
            return
        print(
            f"  AIME eval 2024-2025 exists but only has {len(real)} real "
            f"rows; refetching."
        )

    years = [2024, 2025]
    exams = ["I", "II"]
    fetched = []
    failures = []
    for year in years:
        for exam in exams:
            problems_url = (
                f"https://artofproblemsolving.com/wiki/index.php?title="
                f"{year}_AIME_{exam}_Problems"
            )
            answers_url = (
                f"https://artofproblemsolving.com/wiki/index.php?title="
                f"{year}_AIME_{exam}_Answer_Key"
            )
            try:
                p_html = _aops_fetch(problems_url)
                a_html = _aops_fetch(answers_url)
                problems = _aops_extract_problems(p_html)
                answers = _aops_extract_answer_key(a_html)
                if len(problems) < 15 or len(answers) < 15:
                    raise RuntimeError(
                        f"parsed {len(problems)} problems / {len(answers)} "
                        f"answers for {year} {exam} (expected 15/15)"
                    )
                for (num, text), ans in zip(problems[:15], answers[:15]):
                    fetched.append(_build_aime_train_row(year, exam, num, text, ans))
                print(f"  fetched {year} AIME {exam}: 15 problems")
            except Exception as e:
                failures.append((year, exam, str(e)))
                print(f"  WARN: {year} AIME {exam} failed: {e}")

    if len(fetched) >= 60:
        with open(target, "w", encoding="utf-8") as f:
            for row in fetched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved {len(fetched)} real AIME 2024-2025 problems -> {target}")
        return

    print(
        f"  ERROR: only fetched {len(fetched)}/60 problems; writing a "
        f"PLACEHOLDER eval set so downstream code can still run. "
        f"Re-run with network access to populate real data."
    )
    placeholders = _placeholder_aime_train(years, exams)
    out_rows = fetched + [
        p for p in placeholders
        if not any(
            f["year"] == p["year"] and f["exam"] == p["exam"]
            and f["id"].endswith("_" + p["id"].split("_")[-1])
            for f in fetched
        )
    ]
    with open(target, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"  Wrote {len(out_rows)} rows ({len(fetched)} real, "
        f"{len(out_rows) - len(fetched)} placeholders) -> {target}"
    )
    if failures:
        print("  failures:")
        for y, e, msg in failures:
            print(f"    {y} {e}: {msg}")


def download_arc_agi():
    """Download a text-serialized abstract-reasoning benchmark.

    NOTE on naming: the paper refers to this benchmark slot as 'ARC-AGI-Text'.
    The canonical ARC-AGI corpus (https://github.com/fchollet/ARC-AGI) consists
    of visual-grid puzzles that require a custom text-serialization step.

    This release ships **AI2 ARC-Challenge** (text MCQ science questions from
    `allenai/ai2_arc`) as a *text-only* abstract-reasoning proxy. We use this
    proxy because it provides ~1100 graded text-only items out-of-the-box and
    matches the 'reasoning under text constraints' setting the paper studies.
    Reviewers requiring strict ARC-AGI parity should swap in a serialized
    fchollet/ARC-AGI dump and re-run; the eval harness in
    `cts/eval/arc_agi_text.py` is data-format-agnostic (expects task_id,
    input, output triples).
    """
    out = DATA / "arc_agi"
    out.mkdir(exist_ok=True)
    target = out / "test.jsonl"
    if target.exists():
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  ARC-AGI proxy (ARC-Challenge text MCQ) already exists: {n} problems")
        return

    print("  Downloading ARC-Challenge as text-MCQ proxy for ARC-AGI-Text...")
    _arc_agi_fallback(target)


def _arc_agi_fallback(target: Path):
    """Try ARC challenge dataset as fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        with open(target, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                choices = row.get("choices", {})
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
                item = {
                    "task_id": row.get("id", f"arc_{i}"),
                    "input": row.get("question", "") + "\n" + options,
                    "output": row.get("answerKey", ""),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  Saved {n} ARC-Challenge problems -> {target}")
    except Exception as e2:
        print(f"  ARC fallback also failed: {e2}")


def verify_all():
    """Print summary of all datasets."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    benchmarks = {
        "MATH-500": DATA / "math500" / "test.jsonl",
        "GSM8K": DATA / "gsm8k" / "test.jsonl",
        "HumanEval": DATA / "humaneval" / "test.jsonl",
        "AIME (test 2026)": DATA / "aime" / "test.jsonl",
        "AIME (train 2019-2023)": DATA / "aime" / "train_2019_2023.jsonl",
        "AIME (eval 2024-2025)": DATA / "aime" / "test_2024_2025.jsonl",
        "ARC-AGI": DATA / "arc_agi" / "test.jsonl",
    }
    for name, path in benchmarks.items():
        if path.exists():
            n = sum(1 for l in open(path, "r", encoding="utf-8") if l.strip())
            size_kb = path.stat().st_size / 1024
            print(f"  {name:<12} {n:>6} problems  ({size_kb:.0f} KB)  -> {path}")
        else:
            print(f"  {name:<12}  MISSING  -> {path}")


def download_math500():
    """Download the MATH-500 subset used by paper §6.2 (HendrycksMATH test split)."""
    out = DATA / "math500"
    out.mkdir(exist_ok=True)
    target = out / "test.jsonl"
    if target.exists():
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  MATH-500 already exists: {n} problems")
        return
    print("  Downloading MATH-500 (HuggingFaceH4/MATH-500)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        with open(target, "w", encoding="utf-8") as f:
            for row in ds:
                item = {
                    "problem": row.get("problem", ""),
                    "answer": str(row.get("answer", "")),
                    "subject": row.get("subject", ""),
                    "level": row.get("level", ""),
                    "unique_id": row.get("unique_id", ""),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {len(ds)} problems -> {target}")
    except Exception as e:
        print(f"  MATH-500 download failed: {e}")
        print("  Manual fallback: clone https://huggingface.co/datasets/HuggingFaceH4/MATH-500")


def download_gsm8k():
    """Download GSM8K test split used by paper §6.2."""
    out = DATA / "gsm8k"
    out.mkdir(exist_ok=True)
    target = out / "test.jsonl"
    if target.exists():
        n = sum(1 for l in open(target, "r", encoding="utf-8") if l.strip())
        print(f"  GSM8K already exists: {n} problems")
        return
    print("  Downloading GSM8K (openai/gsm8k 'main' split=test)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        with open(target, "w", encoding="utf-8") as f:
            for row in ds:
                item = {
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {len(ds)} problems -> {target}")
    except Exception as e:
        print(f"  GSM8K download failed: {e}")
        print("  Manual fallback: clone https://huggingface.co/datasets/openai/gsm8k")


def main():
    print("=" * 60)
    print("Downloading All Benchmark Datasets (paper §6.2)")
    print("=" * 60)
    print(
        "Note: For NeurIPS 2026 reproducibility we recommend pinning the "
        "Hugging Face dataset revisions reported in REPRODUCIBILITY.md."
    )

    print("\n[1/5] MATH-500")
    download_math500()

    print("\n[2/5] GSM8K")
    download_gsm8k()

    print("\n[3/5] HumanEval")
    download_humaneval()

    print("\n[4/7] AIME 2026 (paper-aligned, 30 problems)")
    download_aime()

    print("\n[5/7] AIME 2019-2023 train pool (Stage 2 PPO, 150 problems)")
    download_aime_train_2019_2023()

    print("\n[6/7] AIME 2024-2025 eval extension (Table 17, 60 problems)")
    download_aime_eval_2024_2025()

    print("\n[7/7] ARC-AGI proxy (ARC-Challenge text MCQ)")
    download_arc_agi()

    verify_all()


if __name__ == "__main__":
    main()
