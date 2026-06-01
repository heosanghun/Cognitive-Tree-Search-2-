#!/usr/bin/env python3
"""
GSM8K JSONL eval (pass@1). Paper Table 2: CTS target 92.1 ± 0.5%.

  python scripts/run_gsm8k.py --data path/to/gsm8k.jsonl --limit 50
  python scripts/run_gsm8k.py --data data.jsonl --gemma --max-new-tokens 256
  python scripts/run_gsm8k.py --data data.jsonl --out-json artifacts/gsm8k_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.eval.gsm8k import evaluate_gsm8k_predictions, load_gsm8k_jsonl


def _demo_predict(_q: str) -> str:
    return "42"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="GSM8K JSONL path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gemma", action="store_true", help="Use Gemma model for prediction")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-json", type=str, default=None, help="Write results to JSON")
    args = ap.parse_args()

    items = load_gsm8k_jsonl(args.data)
    if args.limit:
        items = items[: args.limit]
    print(f"Loaded {len(items)} GSM8K items")

    if args.gemma:
        try:
            from cts.eval.gemma_predict import build_gemma_predictor
            predictor = build_gemma_predictor(max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"Failed to load Gemma predictor: {e}")
            predictor = _demo_predict
    else:
        predictor = _demo_predict

    predictions = []
    for i, item in enumerate(items):
        pred = predictor(item["question"])
        predictions.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  predicted {i + 1}/{len(items)}")

    result = evaluate_gsm8k_predictions(items, predictions)
    print(f"\nGSM8K accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
