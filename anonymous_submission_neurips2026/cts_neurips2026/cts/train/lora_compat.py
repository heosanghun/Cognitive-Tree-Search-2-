"""In-process LoRA adapter compatible with transformers 5.x + Gemma 4.

Why this exists
---------------

The paper (§6.1) trains a LoRA adapter (r=8, α=16, dropout=0.05) on
``q_proj``, ``v_proj`` and ``o_proj`` of every Gemma 4 decoder block. The
canonical implementation calls ``peft.get_peft_model``, but at the time
of the NeurIPS 2026 submission no released version of ``peft`` is
compatible with the transformers release that ships
``Gemma4ForConditionalGeneration``:

* peft 0.17.x raises ``ImportError: cannot import name 'HybridCache'``
  on transformers >=5 (HybridCache was removed when Gemma 4 landed).
* peft 0.19.1 imports cleanly but ``get_peft_model`` raises
  ``AttributeError: 'LoraModel' object has no attribute
  'prepare_inputs_for_generation'`` because the base-model methods it
  inherits were renamed/removed in transformers 5.x.

Both errors are upstream-fix territory and are blocking on a peft
release that lags the transformers cadence; waiting is not compatible
with the submission deadline.

This module reproduces the exact LoRA math used by peft
(``LoraConfig(r, lora_alpha=2*r, lora_dropout=0.05,
target_modules=...)`` with ``bias="none"``) by walking the model and
substituting matching ``nn.Linear`` children with :class:`LoraLinear`.
The resulting state-dict keys (``lora_A.weight``, ``lora_B.weight``,
``base.weight``) round-trip cleanly between Stage 1 (which trains
LoRA) and Stage 2 / inference (which load the adapter).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LoraLinear(nn.Module):
    """Frozen ``nn.Linear`` augmented with a low-rank A/B residual.

    Forward: ``y = base(x) + (B @ A @ dropout(x)) * (alpha / r)``.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_A = nn.Linear(in_f, rank, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(rank, out_f, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.base(x)
        a = self.lora_A(self.lora_dropout(x))
        b = self.lora_B(a) * self.scaling
        return out + b


def replace_target_linears(
    module: nn.Module,
    target_names: Iterable[str],
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[int, int]:
    """Recursively swap ``nn.Linear`` children whose attribute name is in
    ``target_names`` with :class:`LoraLinear`. Returns
    ``(n_newly_wrapped, n_already_wrapped)`` so the caller can tell apart
    a fresh application from an idempotent re-call (eval-time cached
    backbone scenario: AIME loads the model + LoRA, HumanEval reuses
    the cached backbone and re-enters this routine -- the second call
    must NOT raise just because the modules are already wrapped).
    """
    target_set = set(target_names)
    n_new = 0
    n_already = 0
    for name, child in list(module.named_children()):
        if name in target_set and isinstance(child, nn.Linear):
            wrapped = LoraLinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, wrapped)
            n_new += 1
        elif name in target_set and isinstance(child, LoraLinear):
            n_already += 1
        else:
            sub_new, sub_already = replace_target_linears(
                child, target_set, rank=rank, alpha=alpha, dropout=dropout
            )
            n_new += sub_new
            n_already += sub_already
    return n_new, n_already


def apply_paper_lora(
    backbone,
    *,
    rank: int = 8,
    target_modules: Iterable[str] = ("q_proj", "v_proj", "o_proj"),
    dropout: float = 0.05,
    require_match: bool = True,
    verbose: bool = True,
):
    """Paper §6.1: LoRA r=8, α=16 on q/v/o_proj of the language model.

    ``backbone`` must be a :class:`cts.backbone.gemma_adapter.GemmaCTSBackbone`
    or any object exposing ``cg.model.language_model``. The adapter is
    applied *in place*; the same backbone is returned for chaining.

    Parameters
    ----------
    rank : int
        LoRA rank ``r``. The peft-equivalent ``alpha`` is ``2 * rank``
        (paper convention; matches ``LoraConfig(r=rank, lora_alpha=2*rank)``).
    target_modules : iterable of str
        Linear-layer attribute names to wrap.
    dropout : float
        Dropout applied inside the LoRA branch (paper §6.1: 0.05).
    require_match : bool
        If True (default), raise when no nn.Linear matched. Stage 1 must
        not silently degrade to a frozen backbone; eval / Stage 2 may
        loosen this but Stage 1 specifically demands ``True``.
    verbose : bool
        Print a one-line summary of how many modules were wrapped.
    """
    lm = backbone.cg.model.language_model
    n_new, n_already = replace_target_linears(
        lm, target_modules, rank=rank, alpha=rank * 2, dropout=dropout
    )
    n_total = n_new + n_already
    if n_total == 0 and require_match:
        raise RuntimeError(
            f"apply_paper_lora: no nn.Linear (or pre-wrapped LoraLinear) "
            f"children matched target_modules={list(target_modules)}; "
            f"check transformers version (Gemma 4 q_proj/v_proj/o_proj "
            f"attribute names) and the {type(lm).__name__} module structure."
        )
    if verbose:
        if n_new > 0 and n_already == 0:
            print(
                f"[lora] manual LoRA r={rank} alpha={rank * 2} applied to "
                f"{n_new} nn.Linear modules in {type(lm).__name__}"
            )
        elif n_already > 0 and n_new == 0:
            print(
                f"[lora] LoRA already installed: {n_already} LoraLinear "
                f"modules detected in {type(lm).__name__}; skipping re-wrap "
                f"(idempotent re-entry)."
            )
        else:
            print(
                f"[lora] mixed state: wrapped {n_new} fresh nn.Linear + "
                f"detected {n_already} pre-existing LoraLinear in "
                f"{type(lm).__name__}"
            )
    return backbone


__all__ = [
    "LoraLinear",
    "replace_target_linears",
    "apply_paper_lora",
]
