#!/usr/bin/env python3
"""Honest Hybrid-KV (paper §7.7) measurement CLI.

Usage:

    python scripts/measure_hybrid_kv.py \
        --problems data/aime/test.jsonl \
        --limit 10 \
        --seeds 0 1 2 \
        --out results/hybrid_kv/measurement.md

Defaults to a CPU-friendly mock backbone so CI and reviewer-quick-verify
work without a GPU. The script is idempotent (rewrites the Markdown
report and JSONL trace on every run) and consistent with the README's
``⚠️ decision-plumbed; KV-reuse pending`` disclosure: no number in the
output report claims to measure the cache-HIT path.

Outputs:
  - Markdown report to ``--out`` (the file path the user passes).
  - JSONL trace to ``--jsonl`` (default: alongside the markdown, suffix
    ``.jsonl``).
  - Summary JSON to ``--summary-json`` (default: alongside the markdown,
    suffix ``.summary.json``) so a notebook / dashboard can parse it
    without re-running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from cts.backbone.mock_tiny import MockTinyBackbone  # noqa: E402
from cts.critic.neuro_critic import NeuroCritic  # noqa: E402
from cts.eval.hybrid_kv_measurement import (  # noqa: E402
    measure_decision_overhead,
    render_hybrid_kv_markdown,
    summarize_hybrid_kv,
    write_trace_jsonl,
)
from cts.policy.meta_policy import MetaPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Mock backbone with deterministic z*->text decoder (matches the test suite)
# ---------------------------------------------------------------------------

class _DecodingMockBackbone(MockTinyBackbone):
    """Same decoder pattern as ``tests/test_cts_full_episode.py`` so the
    CLI exercises identical code paths to the regression tests."""

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 64) -> str:
        head = z_star.detach().float().mean().item()
        return f"answer={head:+.4f}|tokens={max_new_tokens}"


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

_SYNTHETIC_FALLBACK: List[Dict[str, str]] = [
    {"problem": f"Q{i}: {i} + {i + 1} = ?", "answer": str(2 * i + 1)} for i in range(8)
]


def _load_problems(path: Path | None, limit: int) -> List[Dict[str, Any]]:
    """Load ``limit`` problems from a JSONL file. Falls back to a small
    synthetic prompt list if the file is missing — keeps the CLI usable
    on a fresh checkout (CI, reviewer quick-verify) without the AIME
    JSONL on disk.
    """
    if path is None or not Path(path).is_file():
        items = _SYNTHETIC_FALLBACK
    else:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    items.append({"problem": line})
    if limit > 0:
        items = items[:limit]
    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Honest Hybrid-KV (paper §7.7) decision-overhead measurement. "
            "CPU-only by default. Writes a Markdown report whose first "
            "30 lines explicitly disclose that the KV-reuse hit path is "
            "NOT YET measured (consistent with the README)."
        )
    )
    p.add_argument(
        "--problems",
        type=Path,
        default=ROOT / "data" / "aime" / "test.jsonl",
        help=(
            "JSONL of problems with a ``problem`` (or ``prompt``) field. "
            "Falls back to a small synthetic prompt list if the file is "
            "absent so CI / reviewer quick-verify still work."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=10,
        help="Max number of problems to use (default: 10).",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Seeds to sweep (default: 0 1 2).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "hybrid_kv" / "measurement.md",
        help="Markdown report path (idempotent).",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=(
            "Long-form per-cell JSONL trace path. Defaults to "
            "``<out>.jsonl``."
        ),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Optional summary JSON path. Defaults to "
            "``<out>.summary.json``."
        ),
    )
    p.add_argument(
        "--hidden", type=int, default=16,
        help="Mock backbone hidden size (default: 16, CPU-friendly).",
    )
    p.add_argument(
        "--num-layers", type=int, default=4,
        help="Mock backbone layer count (default: 4).",
    )
    p.add_argument(
        "--margin-frac", type=float, default=0.05,
        help="TOST equivalence margin as fraction of hybrid_off mean (default: 0.05).",
    )
    p.add_argument(
        "--alpha", type=float, default=0.05,
        help="TOST significance level (default: 0.05).",
    )
    p.add_argument(
        "--W", type=int, default=2, help="Branching factor (default: 2).",
    )
    p.add_argument(
        "--K", type=int, default=4, help="Latent rollout count (default: 4).",
    )
    p.add_argument(
        "--max-decode-tokens", type=int, default=4,
        help="Decoder max tokens (default: 4, CPU-friendly).",
    )
    p.add_argument(
        "--wall-clock-budget-s", type=float, default=10.0,
        help="Per-episode wall-clock budget in seconds (default: 10.0).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress stdout summary (still writes all output files).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    md_path: Path = args.out
    jsonl_path: Path = args.jsonl or md_path.with_suffix(".jsonl")
    summary_json_path: Path = args.summary_json or md_path.with_suffix(".summary.json")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    problems = _load_problems(args.problems, limit=args.limit)
    if not problems:
        print("ERROR: no problems loaded; aborting.", file=sys.stderr)
        return 2

    torch.manual_seed(2026)
    bb = _DecodingMockBackbone(hidden=args.hidden, num_layers=args.num_layers)
    meta = MetaPolicy(text_dim=args.hidden, hidden=32, W=args.W)
    critic = NeuroCritic(z_dim=args.hidden)

    df = measure_decision_overhead(
        bb,
        problems,
        seeds=list(args.seeds),
        meta_policy=meta,
        critic=critic,
        K=args.K,
        W=args.W,
        max_decode_tokens=args.max_decode_tokens,
        wall_clock_budget_s=args.wall_clock_budget_s,
    )
    write_trace_jsonl(df, jsonl_path)

    summary = summarize_hybrid_kv(df, margin_frac=args.margin_frac, alpha=args.alpha)
    render_hybrid_kv_markdown(summary, md_path)

    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    if not args.quiet:
        tost = summary.get("tost", {})
        off = summary.get("by_mode", {}).get("hybrid_off", {})
        on = summary.get("by_mode", {}).get("hybrid_decision_only", {})
        print(
            "[measure_hybrid_kv] DISCLOSURE: KV-reuse hit path NOT YET measured.\n"
            f"  rows: {len(df)}\n"
            f"  hybrid_off:            {off.get('wall_seconds_mean', 0.0):.4f} ± "
            f"{off.get('wall_seconds_std', 0.0):.4f} s (n={off.get('n', 0)})\n"
            f"  hybrid_decision_only:  {on.get('wall_seconds_mean', 0.0):.4f} ± "
            f"{on.get('wall_seconds_std', 0.0):.4f} s (n={on.get('n', 0)})\n"
            f"  decision_calls (mean): {on.get('decision_calls_mean', 0.0):.1f}\n"
            f"  cached_nodes (mean):   {on.get('cached_nodes_mean', 0.0):.1f}\n"
            f"  TOST delta:            {tost.get('delta', 0.0):.6f} s\n"
            f"  TOST p_max:            {tost.get('p_max', 1.0):.6f}\n"
            f"  TOST equivalent:       {bool(tost.get('equivalent', False))}\n"
            f"  Markdown -> {md_path}\n"
            f"  JSONL    -> {jsonl_path}\n"
            f"  Summary  -> {summary_json_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
