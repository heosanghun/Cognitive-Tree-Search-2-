#!/usr/bin/env python3
"""Print chat-formatted string using **tokenizer only** (no model.safetensors load)."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.eval.prompt_format import format_user_prompt_chat_string


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="What is 2+2?")
    ap.add_argument("--model-id", type=str, default=None, help="Override; else CTS_GEMMA_MODEL_DIR / default Hub id")
    args = ap.parse_args()
    s = format_user_prompt_chat_string(args.text, model_id=args.model_id)
    print(s)


if __name__ == "__main__":
    main()
