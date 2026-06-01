#!/usr/bin/env python3
"""
HumanEval JSONL eval (pass@1). Paper Table 2: CTS target 74.2 ± 0.6%.

Paper §7.1: evaluated via local, offline execution.
Complies with security_eval.md sandbox policy.

  python scripts/run_humaneval.py --data path/to/humaneval.jsonl --limit 50
  python scripts/run_humaneval.py --data data.jsonl --out-json artifacts/humaneval_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.eval.humaneval import evaluate_humaneval_predictions, load_humaneval_jsonl


def _demo_complete(prompt: str) -> str:
    return "    return 0\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="HumanEval JSONL path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gemma", action="store_true", help="Use Gemma model")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--execute", action="store_true",
                     help="Run code tests (SANDBOX REQUIRED per security_eval.md)")
    ap.add_argument("--out-json", type=str, default=None, help="Write results to JSON")
    args = ap.parse_args()

    items = load_humaneval_jsonl(args.data)
    if args.limit:
        items = items[: args.limit]
    print(f"Loaded {len(items)} HumanEval items")

    if args.gemma:
        try:
            from cts.eval.gemma_predict import build_gemma_predictor
            predictor = build_gemma_predictor(max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"Failed to load Gemma predictor: {e}")
            predictor = _demo_complete
    else:
        predictor = _demo_complete

    completions = []
    for i, item in enumerate(items):
        comp = predictor(item["prompt"])
        completions.append(comp)
        if (i + 1) % 20 == 0:
            print(f"  completed {i + 1}/{len(items)}")

    result = evaluate_humaneval_predictions(
        items, completions, execute=args.execute
    )
    print(f"\nHumanEval accuracy: {result['accuracy']:.4f} "
          f"({result['correct']}/{result['total']}) "
          f"[execute={result['execute_mode']}]")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
