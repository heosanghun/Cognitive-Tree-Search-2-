"""Best-of-N selection scored by Neuro-Critic V(z*) (paper Table 2 BoN@13).

When multiple native-think samples are drawn, the paper ranks them with
the Stage-2 Neuro-Critic rather than a length heuristic.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def bon_select_pred_with_critic(
    *,
    critic: torch.nn.Module,
    backbone,
    raw_candidates: Sequence[str],
    extract_pred_fn,
    benchmark: str,
    device: torch.device,
) -> str:
    """Pick the candidate whose pooled context embedding maximises V_psi.

    ``extract_pred_fn(raw, benchmark) -> str`` is injected so this module
    stays free of ``scripts/`` imports.
    """
    best_pred = ""
    best_score = float("-inf")
    for raw in raw_candidates:
        if not raw or not str(raw).strip():
            continue
        pred = extract_pred_fn(str(raw), benchmark)
        with torch.no_grad():
            ctx = backbone.encode_context(str(raw))
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            score = float(critic(ctx.to(device).float()).squeeze().item())
        if score > best_score:
            best_score = score
            best_pred = pred
    return best_pred


def bon_select_index_with_critic(
    *,
    critic: torch.nn.Module,
    backbone,
    raw_candidates: Sequence[str],
    device: torch.device,
) -> int:
    """Return index of highest Neuro-Critic scored raw completion."""
    best_idx = -1
    best_score = float("-inf")
    for i, raw in enumerate(raw_candidates):
        if not raw or not str(raw).strip():
            continue
        with torch.no_grad():
            ctx = backbone.encode_context(str(raw))
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            score = float(critic(ctx.to(device).float()).squeeze().item())
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx
