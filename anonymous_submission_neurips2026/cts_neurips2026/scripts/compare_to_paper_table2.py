#!/usr/bin/env python3
"""Compare a local Table-2 results JSON to the paper-reported numbers.

Reads ``<results-dir>/table2_results.json`` (produced by
``scripts/run_cts_eval_full.py``) and emits a Markdown comparison table that
shows local mean +/- std with 95% bootstrap CI alongside the paper headline,
plus the absolute gap.

Usage:
    python scripts/compare_to_paper_table2.py results/table2_re1
    python scripts/compare_to_paper_table2.py results/table2_re1 --out compare.md

Designed to be run after a re-run completes; safe to invoke even on partial
results (missing cells render as ``--``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# Paper Table 2 (NeurIPS submission, Gemma 4 E4B, 5 seeds, <= 1e14 MACs).
# Mean accuracy (%) on each benchmark. ``None`` means the paper does not
# report that combination.
PAPER_TABLE2: Dict[str, Dict[str, Optional[float]]] = {
    "greedy":            {"math500": 45.2, "gsm8k": 76.5, "aime": 28.3, "arc_agi_text": 36.1, "humaneval": 56.4},
    "sc_14":             {"math500": 59.3, "gsm8k": 84.2, "aime": 34.8, "arc_agi_text": 52.4, "humaneval": 65.2},
    "native_think":      {"math500": 57.0, "gsm8k": 82.4, "aime": 42.5, "arc_agi_text": 50.1, "humaneval": 63.3},
    "mcts_early_stop":   {"math500": 56.5, "gsm8k": 81.2, "aime": 38.4, "arc_agi_text": 48.1, "humaneval": 62.5},
    # NOTE: paper Table 2 cell CTS-4nu / MATH-500 = 64.1 (verified against PDF).
    "cts_4nu":           {"math500": 64.1, "gsm8k": 88.4, "aime": 50.2, "arc_agi_text": 57.8, "humaneval": 69.6},
}

PAPER_STD: Dict[str, Dict[str, Optional[float]]] = {
    "greedy":            {"math500": 0.0,  "gsm8k": 0.0,  "aime": 0.0,  "arc_agi_text": 0.0,  "humaneval": 0.0},
    "sc_14":             {"math500": 0.7,  "gsm8k": 0.5,  "aime": 0.9,  "arc_agi_text": 0.8,  "humaneval": 0.6},
    "native_think":      {"math500": 0.6,  "gsm8k": 0.4,  "aime": 0.9,  "arc_agi_text": 0.7,  "humaneval": 0.5},
    "mcts_early_stop":   {"math500": 0.9,  "gsm8k": 0.7,  "aime": 0.8,  "arc_agi_text": 1.0,  "humaneval": 0.7},
    "cts_4nu":           {"math500": 0.8,  "gsm8k": 0.5,  "aime": 1.1,  "arc_agi_text": 0.9,  "humaneval": 0.7},
}

METHOD_DISPLAY = {
    "greedy":          "Greedy",
    "sc_14":           "SC@14",
    "native_think":    "Native Think",
    "mcts_early_stop": "MCTS (Early Stop)",
    "cts_4nu":         "CTS-4nu (Ours)",
}

BENCHMARK_DISPLAY = {
    "math500":      "MATH-500",
    "gsm8k":        "GSM8K",
    "aime":         "AIME 2026",
    "arc_agi_text": "ARC-AGI-Text",
    "humaneval":    "HumanEval",
}


def _fmt_local(stat: Optional[Dict[str, Any]]) -> str:
    if stat is None:
        return "&mdash;"
    mean = float(stat.get("mean", 0.0)) * 100.0
    std = float(stat.get("std", 0.0)) * 100.0
    n = int(stat.get("n_samples", 0))
    return f"{mean:.1f}&pm;{std:.1f} (n={n})"


def _fmt_paper(method: str, bench: str) -> str:
    mean = PAPER_TABLE2.get(method, {}).get(bench)
    if mean is None:
        return "&mdash;"
    std = PAPER_STD.get(method, {}).get(bench, 0.0) or 0.0
    if std > 0:
        return f"{mean:.1f}&pm;{std:.1f}"
    return f"{mean:.1f}"


def _fmt_gap(stat: Optional[Dict[str, Any]], method: str, bench: str) -> str:
    if stat is None:
        return "&mdash;"
    paper = PAPER_TABLE2.get(method, {}).get(bench)
    if paper is None:
        return "&mdash;"
    local_mean = float(stat.get("mean", 0.0)) * 100.0
    delta = local_mean - paper
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


def render_markdown(local: Dict[str, Dict[str, Any]], source_path: Path) -> str:
    benchmarks = list(BENCHMARK_DISPLAY.keys())
    methods = list(METHOD_DISPLAY.keys())

    lines: list[str] = []
    lines.append(f"# Paper vs Local Comparison &mdash; Table 2")
    lines.append("")
    lines.append(f"Source: `{source_path.as_posix()}`")
    lines.append("")
    lines.append("Each cell shows **local mean&pm;std (n samples)** in the first row and the")
    lines.append("**paper headline** plus the **gap (local &minus; paper)** in subsequent rows.")
    lines.append("All values are accuracy in percent; n is the number of (seed x problem)")
    lines.append("samples that succeeded.")
    lines.append("")

    header_cells = ["Method"] + [BENCHMARK_DISPLAY[b] for b in benchmarks]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

    for method in methods:
        bench_data = local.get(method, {})
        local_row = [METHOD_DISPLAY[method]] + [_fmt_local(bench_data.get(b)) for b in benchmarks]
        paper_row = ["&nbsp;&nbsp;_paper_"] + [_fmt_paper(method, b) for b in benchmarks]
        gap_row   = ["&nbsp;&nbsp;_gap_"]   + [_fmt_gap(bench_data.get(b), method, b) for b in benchmarks]
        lines.append("| " + " | ".join(local_row) + " |")
        lines.append("| " + " | ".join(paper_row) + " |")
        lines.append("| " + " | ".join(gap_row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Paper headlines are taken from Table 2 of the NeurIPS 2026 submission")
    lines.append("  (Gemma 4 E4B backbone, 5 seeds, &le; 1e14 MACs, 95% bootstrap CI).")
    lines.append("- **Baseline coverage disclosure**: the single-GPU snapshot integrates")
    lines.append("  only `greedy`, `native_think`, `cts_2nu`, `cts_4nu`, `deq_only`. Paper")
    lines.append("  baselines `sc_14` and `mcts_early_stop` are rendered above as")
    lines.append("  paper-only reference numbers; the corresponding local rows will read")
    lines.append("  `&mdash;` because `_run_cts_on_problems` raises `NotImplementedError`")
    lines.append("  on those names rather than silently producing greedy-equivalent")
    lines.append("  numbers. See README.md \"Implementation Status\" for full disclosure.")
    lines.append("- The operational primary Bonferroni family in this snapshot is therefore")
    lines.append("  reduced to **n=6** (CTS-4nu vs {greedy, native_think} x")
    lines.append("  {math500, gsm8k, aime}) rather than the paper's n=12. The paper's")
    lines.append("  full n=12 family is reproducible only after the missing baselines are")
    lines.append("  added (multi-GPU paper-scale run).")
    lines.append("- Local re-runs in this repo currently use a reduced wall-clock and MAC")
    lines.append("  budget (CTS_EVAL_TAU_CAP=1e13, CTS_EVAL_EPISODE_TIMEOUT=180s) so")
    lines.append("  absolute accuracy is expected to be lower than the paper headline; the")
    lines.append("  *relative ordering* across methods is the headline reproducibility")
    lines.append("  signal.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("results_dir", help="Directory containing table2_results.json")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write the rendered Markdown (defaults to <results_dir>/PAPER_VS_LOCAL.md)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_path = results_dir / "table2_results.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found", file=sys.stderr)
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        local = json.load(f)

    md = render_markdown(local, json_path)

    out_path = Path(args.out) if args.out else (results_dir / "PAPER_VS_LOCAL.md")

    # Preserve any reviewer-facing appendix (e.g. the "Why the Gap?"
    # honest-gap-analysis section added in D11) that lives below an
    # explicit anchor.  The render block above only owns the table +
    # short Notes section; everything from ``GAP_ANALYSIS_ANCHOR``
    # downward is hand-curated documentation that must survive a
    # post-Stage-2 regeneration.
    GAP_ANCHOR = "## Why the Gap?"
    appendix = ""
    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        idx = existing.find(GAP_ANCHOR)
        if idx >= 0:
            appendix = "\n" + existing[idx:]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        if appendix:
            f.write(appendix)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    if appendix:
        print(f"  preserved appendix: {len(appendix.splitlines())} lines below '{GAP_ANCHOR}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
