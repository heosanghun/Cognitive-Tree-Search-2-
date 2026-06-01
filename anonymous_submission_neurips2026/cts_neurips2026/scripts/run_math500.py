#!/usr/bin/env python3
"""
MATH-style JSONL eval (pass@1). Provide `--data` JSONL with `problem`/`answer` fields.

  python scripts/run_math500.py --data path/to/math500.jsonl --limit 50
  python scripts/run_math500.py --data data.jsonl --gemma --max-new-tokens 128
  python scripts/run_math500.py --data data.jsonl --out-json artifacts/math_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.eval.gemma_predict import add_gemma_benchmark_args, build_gemma_predictor
from cts.eval.math500 import evaluate_pass_at_1, load_math_samples
from cts.eval.think_prompt import format_user_prompt_with_thinking


def _demo_predict(_q: str) -> str:
    return "42"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="JSONL path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write full result (summary + per-row items) as UTF-8 JSON",
    )
    ap.add_argument(
        "--think-prompt",
        action="store_true",
        help="Prepend <|think|>-aware chat template string (E4B-it tokenizer; no extra weights)",
    )
    add_gemma_benchmark_args(ap)
    args = ap.parse_args()

    if args.gemma:
        inner = build_gemma_predictor(
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            use_chat_template=args.chat_template,
        )
    else:
        inner = _demo_predict

    if args.think_prompt:

        def predict(q: str) -> str:
            return inner(format_user_prompt_with_thinking(q))

    else:
        predict = inner

    if not args.data:
        print(evaluate_pass_at_1([], predict))
        print("(no --data) stub; provide JSONL for real pass@1)")
        return

    samples = load_math_samples(args.data, limit=args.limit)
    want_items = bool(args.out_json)
    result = evaluate_pass_at_1(samples, predict, include_items=want_items)
    print(result)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script": "run_math500.py",
            "data": os.path.abspath(args.data),
            "limit": args.limit,
            "gemma": bool(args.gemma),
            "think_prompt": bool(args.think_prompt),
            "chat_template": bool(args.chat_template),
            "result": result,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", out)


if __name__ == "__main__":
    main()
