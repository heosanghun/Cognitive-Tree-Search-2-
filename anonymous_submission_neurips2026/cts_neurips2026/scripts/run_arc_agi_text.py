#!/usr/bin/env python3
"""
ARC text-serialized JSONL eval (pass@1). Lines need `input` and `output` (gold).

  python scripts/run_arc_agi_text.py --data path/to/arc.jsonl --limit 100
  python scripts/run_arc_agi_text.py --data arc.jsonl --gemma --max-new-tokens 64
  python scripts/run_arc_agi_text.py --data arc.jsonl --out-json artifacts/arc_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.eval.gemma_predict import add_gemma_benchmark_args, build_gemma_predictor
from cts.eval.arc_agi_text import evaluate_pass_at_1_arc, load_arc_text_samples
from cts.eval.think_prompt import format_user_prompt_with_thinking


def _demo_predict(_inp: str) -> str:
    return "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write full result (summary + per-row items) as UTF-8 JSON",
    )
    ap.add_argument("--think-prompt", action="store_true", help="E4B-it <|think|> template wrap")
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

        def predict(inp: str) -> str:
            return inner(format_user_prompt_with_thinking(inp))

    else:
        predict = inner

    if not args.data:
        print(evaluate_pass_at_1_arc([], predict))
        print("(no --data) stub; provide JSONL")
        return

    samples = load_arc_text_samples(args.data, limit=args.limit)
    want_items = bool(args.out_json)
    result = evaluate_pass_at_1_arc(samples, predict, include_items=want_items)
    print(result)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script": "run_arc_agi_text.py",
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
