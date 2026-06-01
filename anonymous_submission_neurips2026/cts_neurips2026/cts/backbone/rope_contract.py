"""
RoPE / anchor contract for GemmaCTSBackbone (paper: anchor text s_t encoded once).

**Completed separation (API level):**
- **Anchor path:** `encode_context(parent_text)` — tokenizer + full `language_model` forward on prompt
  positions; RoPE is applied by HF as usual for that sequence.
- **Inner path:** `deq_step(z, context, ...)` — operates on latent `z` and pooled `context` without
  re-feeding anchor token IDs through additional LM layers that would duplicate anchor RoPE.

**Optional Phase-2 (not required for DEQ+MCTS pipeline):** custom HF `forward` with explicit
`position_ids` only for anchor length on inner blocks — see `gemma_adapter.py` module docstring.
Set `CTS_ROPE_PHASE2=1` only after HF hook design + tests (future extension).

This module is the **single reference** for “RoPE anchor vs inner z” completion status in CTS.
"""

from __future__ import annotations


def rope_policy_summary() -> str:
    return (
        "Anchor: encode_context (HF RoPE on prompt). "
        "Inner: deq_step on z + context tensor (no duplicate anchor token forward)."
    )


def phase2_custom_forward_available() -> bool:
    """True only when optional HF LM hook is implemented (currently False)."""
    return False
