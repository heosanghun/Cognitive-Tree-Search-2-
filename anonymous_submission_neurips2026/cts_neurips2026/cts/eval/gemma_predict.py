"""
Greedy text generation with Gemma 4 E4B for MATH / ARC bench scripts.

Uses `load_gemma4_e4b` and `model.generate` (not CTS DEQ). For Iso-FLOP accounting,
DEQ inner solve is separate from this **decode-only** LM forward (see `configs/README.md`).
"""

from __future__ import annotations

import argparse
from typing import Callable, Optional

import torch

from cts.model.gemma_loader import DEFAULT_GEMMA4_E4B_ID, load_gemma4_e4b


class GemmaTextPredictor:
    """Callable `str -> str`: greedy continuation after `prompt` (raw or chat-templated)."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        *,
        max_new_tokens: int = 256,
        device: Optional[torch.device | str] = None,
        use_chat_template: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template
        if device is not None:
            self._device = torch.device(device)
        else:
            p = next(model.parameters())
            self._device = p.device

    def _pad_id(self) -> int:
        tid = getattr(self.tokenizer, "pad_token_id", None)
        if tid is None:
            tid = getattr(self.tokenizer, "eos_token_id", None)
        if tid is None:
            return 0
        return int(tid)

    @torch.inference_mode()
    def __call__(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        """Greedy decode `prompt`. Per-call ``max_new_tokens`` overrides the
        instance default; this is critical for short-answer benchmarks
        (ARC-AGI-Text MCQ, AIME integer answers) where decoding the full
        ``self.max_new_tokens`` budget can otherwise dominate wall-clock and
        starve the wall-clock-budgeted MCTS loop.
        """
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            raw = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(raw, torch.Tensor):
                input_ids = raw.to(self._device)
            else:
                input_ids = raw["input_ids"].to(self._device)
            attn = torch.ones_like(input_ids, dtype=torch.long, device=self._device)
        else:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            input_ids = enc["input_ids"].to(self._device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(self._device)
        pad = self._pad_id()
        n_new = int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=n_new,
            do_sample=False,
            pad_token_id=pad,
        )
        in_len = input_ids.shape[1]
        new_tokens = gen[0, in_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_gemma_predictor(
    *,
    model_id: Optional[str] = None,
    max_new_tokens: int = 256,
    device_map: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    use_chat_template: bool = False,
) -> Callable[[str], str]:
    """
    Load Gemma once and return `predict(prompt) -> completion`.

    `CTS_GEMMA_MODEL_DIR` is respected when using default Hub id (see `load_gemma4_e4b`).
    """
    dm = device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mid = model_id or DEFAULT_GEMMA4_E4B_ID
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=dm,
        torch_dtype=torch_dtype,
    )
    dev = next(model.parameters()).device
    return GemmaTextPredictor(
        model,
        tok,
        max_new_tokens=max_new_tokens,
        device=dev,
        use_chat_template=use_chat_template,
    )


def add_gemma_benchmark_args(ap: argparse.ArgumentParser) -> None:
    """Shared flags for `run_math500.py` / `run_arc_agi_text.py`."""
    ap.add_argument(
        "--gemma",
        action="store_true",
        help="Use Gemma 4 E4B greedy generate (needs weights; see CTS_GEMMA_MODEL_DIR)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="HF device_map (default: cuda:0 if available else cpu)",
    )
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap prompt with tokenizer.apply_chat_template (E4B-it / chat models)",
    )
