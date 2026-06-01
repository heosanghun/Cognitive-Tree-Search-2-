"""Stage 1: DEQ warm-up — IFT residual loss + LM preservation (paper §6).

Paper §6: loss = ||f(z*) - z*||^2 + 0.1 * L_CE

The L_CE term limits perplexity increase to 0.4 nats (vs. 1.8 nats without it;
Appendix P). Without it, DEQ-Only AIME drops from 35.2% to 31.4%.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.types import NuVector


def fixed_point_surrogate_loss(
    backbone: nn.Module,
    parent_text: str,
    z: torch.Tensor,
    nu: NuVector,
    *,
    w_g: torch.Tensor,
    top_k: int = 3,
    extra: Dict[str, Any] | None = None,
    lambda_lm: float = 0.1,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Paper §6: ||Phi(z) - z||^2 + lambda_lm * L_CE.

    The IFT residual drives z toward a fixed point.
    The CE term preserves the language model's generative capacity.
    """
    device = z.device
    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)
    zf = z.float()
    alpha = routing_weights(zf, w_g.to(zf.device), nu.nu_temp)
    mw = sparse_module_weights(alpha, top_k)
    ex = dict(extra or {})
    phi_z = backbone.deq_step(z, context, mw, ex)

    loss_fp = F.mse_loss(phi_z.float(), z.float())

    loss_ce = torch.zeros((), device=device)
    if lambda_lm > 0.0 and tokenizer is not None and hasattr(backbone, "cg"):
        try:
            enc = tokenizer(
                parent_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.set_grad_enabled(backbone.training):
                lm_out = backbone.cg(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_cache=False,
                )
            if hasattr(lm_out, "loss") and lm_out.loss is not None:
                loss_ce = lm_out.loss
        except Exception:
            pass

    return loss_fp + lambda_lm * loss_ce


def run_stage1_demo_step(
    *,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> Tuple[float, MockTinyBackbone]:
    """One Adam step on `MockTinyBackbone`; returns (loss_value, backbone)."""
    dev = device or torch.device("cpu")
    bb = MockTinyBackbone(hidden=64, num_layers=42).to(dev)
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
    w_g = torch.randn(19, 64, device=dev) * 0.02
    z = torch.randn(8, 64, device=dev)
    opt.zero_grad()
    loss = fixed_point_surrogate_loss(bb, "demo prompt", z, nu, w_g=w_g)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu().item()), bb
