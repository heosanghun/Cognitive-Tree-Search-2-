#!/usr/bin/env python3
"""Run the AIME train/test contamination screen and exit non-zero on FAIL.

Default invocation matches the paper §7.1 audit:

    python scripts/run_contamination_screen.py \
        --train data/aime/train_2019_2023.jsonl \
        --test  data/aime/test.jsonl \
        --out   results/contamination/aime_screen.md

The script is idempotent (rewrites the report on every run).  Exit codes:

* ``0`` -- ``PASS`` or ``WARN`` (lexical overlap only, no MinHash near-dup).
* ``1`` -- ``FAIL`` (MinHash near-duplicate hit -- AIME headline number is
  invalidated; CI must block on this).
* ``2`` -- file-not-found / argument errors.

This matches the verdict policy in :func:`cts.data.contamination_screen.screen_aime_train_test`:
MinHash is the binding contamination gate, BM25 is a topical-overlap signal
that warrants reviewer inspection but does NOT invalidate the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cts.data.contamination_screen import screen_aime_train_test


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the AIME train/test contamination screen and write a "
            "Markdown verdict. Exits 1 if any pair is flagged."
        )
    )
    p.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "aime" / "train_2019_2023.jsonl",
        help="JSONL of train problems (default: data/aime/train_2019_2023.jsonl)",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=ROOT / "data" / "aime" / "test.jsonl",
        help="JSONL of held-out test problems (default: data/aime/test.jsonl)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "contamination" / "aime_screen.md",
        help="Markdown report path",
    )
    p.add_argument(
        "--bm25-flag-threshold",
        type=float,
        default=0.5,
        help="Normalised BM25 score above which a pair is flagged (default: 0.5)",
    )
    p.add_argument(
        "--bm25-top-k",
        type=int,
        default=5,
        help="Number of train matches reported per test item (default: 5)",
    )
    p.add_argument(
        "--minhash-threshold",
        type=float,
        default=0.8,
        help="MinHash Jaccard threshold (default: 0.8)",
    )
    p.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Also print the screen result as JSON to stdout (alongside the Markdown)",
    )
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.train.exists():
        print(f"ERROR: train file not found: {args.train}", file=sys.stderr)
        return 2
    if not args.test.exists():
        print(f"ERROR: test file not found: {args.test}", file=sys.stderr)
        return 2

    result = screen_aime_train_test(
        train_jsonl=args.train,
        test_jsonl=args.test,
        output_md=args.out,
        bm25_flag_threshold=args.bm25_flag_threshold,
        bm25_top_k=args.bm25_top_k,
        minhash_threshold=args.minhash_threshold,
        num_perm=args.num_perm,
    )

    print(
        f"[contamination] verdict={result['verdict']} "
        f"sub_verdict={result['sub_verdict']} "
        f"n_train={result['n_train']} n_test={result['n_test']} "
        f"bm25_flagged={len(result['bm25_flagged'])} "
        f"minhash_flagged={len(result['minhash_flagged'])} "
        f"report={result['report_path']}"
    )
    if result["verdict"] == "WARN":
        print(
            "[contamination] WARN: BM25 lexical-overlap detected but no MinHash "
            "near-duplicate. Review the flagged pairs in the Markdown report; "
            "this does NOT invalidate the held-out test split.",
            file=sys.stderr,
        )
    if args.json:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()
    # FAIL is the binding gate (MinHash near-dup); WARN/PASS exit 0.
    return 1 if result["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
