"""Unit test for the per-call max_new_tokens override on GemmaTextPredictor.

The fix to address the ARC-AGI-Text hang depends on
``GemmaTextPredictor.__call__`` honoring a per-call ``max_new_tokens``
keyword argument so that short-answer benchmarks (MCQ letter, integer
answer) can decode 8-32 tokens instead of the 512-token instance default.

This test uses a tiny stand-in for ``model`` and ``tokenizer`` (no
Hugging Face download, no GPU) and asserts that:

  1. ``predictor(prompt)`` continues to use the instance default;
  2. ``predictor(prompt, max_new_tokens=N)`` forwards ``N`` directly to
     ``model.generate``;
  3. the override is keyword-only (positional misuse raises TypeError).
"""

from __future__ import annotations

import torch

from cts.eval.gemma_predict import GemmaTextPredictor


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=4096):
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return "x" * int(ids.numel())


class _MockModel(torch.nn.Module):
    """A throwaway nn.Module that records the ``max_new_tokens`` it was called with."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.last_max_new_tokens: int = -1

    def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample, pad_token_id):
        self.last_max_new_tokens = int(max_new_tokens)
        new = torch.zeros(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


def _build_predictor(default_max: int) -> tuple[GemmaTextPredictor, _MockModel]:
    model = _MockModel()
    tok = _MockTokenizer()
    pred = GemmaTextPredictor(model, tok, max_new_tokens=default_max, device=torch.device("cpu"))
    return pred, model


def test_predictor_uses_instance_default_when_no_override():
    pred, model = _build_predictor(default_max=512)
    _ = pred("hello")
    assert model.last_max_new_tokens == 512


def test_predictor_honors_per_call_override():
    pred, model = _build_predictor(default_max=512)
    _ = pred("hello", max_new_tokens=8)
    assert model.last_max_new_tokens == 8


def test_predictor_per_call_override_is_isolated():
    pred, model = _build_predictor(default_max=128)
    _ = pred("a", max_new_tokens=4)
    assert model.last_max_new_tokens == 4
    _ = pred("b")  # falls back to the instance default
    assert model.last_max_new_tokens == 128


def test_predictor_override_is_keyword_only():
    pred, _ = _build_predictor(default_max=64)
    try:
        pred("hello", 8)  # noqa - intentionally positional
    except TypeError:
        return
    raise AssertionError("max_new_tokens override must be keyword-only")
