"""Stage-2 PPO rollout reward helpers (paper Eq. 5).

Paper Eq.(5): R_total = 1{correct_answer} - lambda_halt * T.

The shipped JSONL historically carried prompts only (no gold solution).
When ``solution`` / ``answer`` is absent, the trainer falls back to a
**DEQ convergence proxy** (``solver_stats['converged']``) so PPO can
still run without an oracle. When gold is present *and*
``stage2_rollout_decode_tokens`` is large enough for ``child_text`` to
contain an extractable answer, the answer-based path is used instead.

See ``LIMITATIONS.md`` §15 for the honest disclosure.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cts.rewards.shaping import paper_reward


def gold_from_stage2_row(row: Dict[str, Any]) -> Optional[str]:
    """Extract gold solution text from a Stage-2 JSONL row, if present."""
    if not isinstance(row, dict):
        return None
    for key in ("solution", "answer"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def stage2_rollout_is_correct(
    row: Dict[str, Any],
    child_text: Optional[str],
    *,
    converged: bool,
    reward_mode: str = "auto",
) -> bool:
    """Decide the ``correct`` bit for ``paper_reward()`` (Eq. 5).

    Modes (``configs/default.yaml`` key ``stage2_reward_mode``):

    - ``converged``: legacy proxy — Broyden fixed-point convergence only.
    - ``answer``: gold required; False when gold or ``child_text`` missing.
    - ``auto`` (default): gold + decodable ``child_text`` → answer match;
      otherwise convergence proxy.
    """
    mode = (reward_mode or "auto").strip().lower()
    if mode == "converged":
        return bool(converged)

    gold = gold_from_stage2_row(row)
    has_oracle = gold is not None and bool((child_text or "").strip())

    if mode == "answer":
        if not has_oracle:
            return False
        return _answer_matches(child_text or "", gold)

    # auto
    if has_oracle:
        return _answer_matches(child_text or "", gold)
    return bool(converged)


def stage2_rollout_reward(
    row: Dict[str, Any],
    *,
    child_text: Optional[str],
    converged: bool,
    terminal_depth: int,
    lambda_halt: float,
    reward_mode: str = "auto",
) -> float:
    """Paper Eq.(5) reward for one Stage-2 PPO rollout step."""
    correct = stage2_rollout_is_correct(
        row,
        child_text,
        converged=converged,
        reward_mode=reward_mode,
    )
    return paper_reward(
        correct=correct,
        terminal_depth=terminal_depth,
        lambda_halt=lambda_halt,
    )


def _answer_matches(pred_raw: str, gold_raw: str) -> bool:
    from cts.eval.math500 import answers_match, extract_answer, extract_gold

    pred = extract_answer(pred_raw)
    gold = extract_gold(gold_raw)
    if not pred or not gold:
        return False
    return answers_match(pred, gold)
