"""CTS transition API: DEQ fixed point + FAISS context + batch + ACT (paper §4).

Algorithm 1 alignment:
  - Line 6: z_tilde_w = z*_parent + epsilon_w  (parent z* + noise, NOT random init)
  - Line 9: fallback on non-convergence: revert to parent z*, Q <- 0
  - Line 10: L-Broyden inverse Jacobian inheritance via (Us, VTs)
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.broyden_forward import (
    BroydenInfo,
    broyden_fixed_point,
    broyden_fixed_point_batch,
    map_nu_tol_to_tol,
)
from cts.latent.bottleneck import add_exploration_noise, init_z0
from cts.latent.faiss_context import LatentContextWindow, prepend_soft_prefix
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

import os as _os
try:
    from cts.routing.sparse_moe_triton import (
        _TRITON_AVAILABLE as _TRITON_OK,
        routing_weights_triton,
    )
except Exception:
    _TRITON_OK = False
    routing_weights_triton = None  # type: ignore[assignment]

# Set CTS_DISABLE_TRITON=1 to force the PyTorch reference path (e.g. for bit-exact
# debugging, or to compare numerical agreement during a regression run).
_USE_TRITON = _TRITON_OK and (_os.environ.get("CTS_DISABLE_TRITON", "0") != "1")


def _routing_sparse(zz: "torch.Tensor", w_g: "torch.Tensor", nu_temp: float, top_k: int):
    """Single entry point for sparse top-k routing in the DEQ phi hot-path.

    Uses the fused Triton kernel when (a) Triton is importable, (b) tensors live
    on CUDA, and (c) ``CTS_DISABLE_TRITON`` is unset. Otherwise falls back to
    the PyTorch reference (``routing_weights`` + ``sparse_module_weights``),
    which is unit-tested for numerical equivalence in
    ``tests/test_routing_triton_ref.py``.
    """
    if _USE_TRITON and zz.is_cuda and routing_weights_triton is not None:
        try:
            return routing_weights_triton(zz, w_g, nu_temp, top_k=top_k)
        except Exception:
            pass
    alpha = routing_weights(zz, w_g, nu_temp)
    return sparse_module_weights(alpha, top_k)
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


def _load_mac_lut() -> list:
    p = Path(__file__).resolve().parent.parent / "routing" / "lut_mac.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["mac_per_module"])


def transition(
    parent_text: str,
    branch_index: int,
    nu: NuVector,
    budget: RuntimeBudgetState,
    backbone: BaseCTSBackbone,
    *,
    K: int = 64,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e14,
    generator: Optional[torch.Generator] = None,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
    faiss_context: Optional[LatentContextWindow] = None,
    fp32_buffer: bool = True,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    parent_z_star: Optional[torch.Tensor] = None,
    noise_sigma: float = 0.02,
) -> TransitionResult:
    """One KV-cache-free transition using DEQ (paper §4.2).

    Paper Algorithm 1 alignment:
      Line 6:  {z_tilde_w} <- z*_s + epsilon_w
      Line 9:  fallback: revert to parent z*, Q <- 0
      Line 10: L-Broyden-Update inherits parent inverse Jacobian
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d = d or backbone.hidden_size
    gen = generator or torch.Generator(device=device)
    gen.manual_seed(2026 + branch_index * 31 + (len(parent_text) % 997))

    # Algorithm 1 line 6: z_tilde_w = z*_parent + epsilon_w
    if parent_z_star is not None:
        pz = parent_z_star.detach().to(device=device, dtype=torch.float32)
        if pz.shape[0] != K or pz.shape[-1] != d:
            pz = init_z0(K, d, device, gen)
        epsilon = torch.randn(K, d, device=device, dtype=torch.float32, generator=gen) * noise_sigma * nu.nu_expl
        z0 = pz + epsilon
    else:
        z0 = init_z0(K, d, device, gen)
        z0 = add_exploration_noise(z0, nu.nu_expl, gen)

    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)

    faiss_retrieved = None
    if faiss_context is not None:
        faiss_retrieved_raw = faiss_context.retrieve(z0, k=3)
        if faiss_retrieved_raw is not None:
            context = prepend_soft_prefix(context, faiss_retrieved_raw)
            faiss_retrieved = faiss_retrieved_raw

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()

    def phi(zz: torch.Tensor) -> torch.Tensor:
        if routing_mode == "dense":
            mw = routing_weights(zz, w_g, nu.nu_temp)
        else:
            mw = _routing_sparse(zz, w_g, nu.nu_temp, top_k)
        extra: Dict[str, Any] = {"top_k": top_k}
        return backbone.deq_step(zz, context, mw, extra)

    tol = map_nu_tol_to_tol(nu.nu_tol, broyden_tol_min, broyden_tol_max)

    inherited_jac = None
    if parent_inv_jacobian is not None:
        inherited_jac = parent_inv_jacobian

    z_star, info = broyden_fixed_point(
        phi, z0, tol=tol, max_iter=broyden_max_iter, fp32_buffer=fp32_buffer,
        parent_inv_jacobian=inherited_jac,
    )

    budget = budget.clone()
    budget.terminal_depth += 1
    flops = 0.0
    with torch.no_grad():
        alpha = routing_weights(z_star, w_g, nu.nu_temp)
        mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
        for i in range(19):
            flops += float(mw[i].item()) * macs[i] * nu.nu_act
    budget.flops_spent_step = flops
    budget.mac_accumulated += flops

    phi_evals_per_broyden_iter = 2
    flops_broyden_estimate = (
        flops * float(info.iterations) * float(phi_evals_per_broyden_iter)
    )

    solver_stats: Dict[str, Any] = {
        "iterations": info.iterations,
        "residual_norm": info.residual_norm,
        "converged": info.converged,
        "flops_used": flops,
        "flops_inner_once": flops,
        "flops_broyden_estimate": flops_broyden_estimate,
        "phi_evals_per_broyden_iter": phi_evals_per_broyden_iter,
        # Paper Remark 2: expose the converged inverse Jacobian so the MCTS
        # episode loop can thread it into child transitions. None on the
        # Anderson path (large-n / Gemma-scale tensors).
        "inv_jacobian": getattr(info, "jacobian_state", None),
    }

    # Algorithm 1 line 9: fallback on non-convergence → revert to parent z*, Q <- 0
    if not info.converged:
        fallback_z = parent_z_star.detach().clone().to(device) if parent_z_star is not None else z_star
        return TransitionResult(
            child_text=None,
            z_star_child=fallback_z,
            solver_stats=solver_stats,
            prune=True,
            budget=budget,
            faiss_retrieved=faiss_retrieved,
        )

    if budget.mac_accumulated > tau_flops_budget * nu.nu_act:
        solver_stats["act_halt"] = True

    if faiss_context is not None:
        faiss_context.add(z_star)

    if hasattr(backbone, "decode_from_z_star"):
        try:
            dec = getattr(backbone, "decode_from_z_star")
            sig = inspect.signature(dec)
            if "max_new_tokens" in sig.parameters:
                child_text = dec(z_star, max_new_tokens=max_decode_tokens)
            else:
                child_text = dec(z_star)
        except Exception:
            flat = z_star.detach().cpu().reshape(-1)[:16]
            h = int(torch.sum(flat * 1e4).item()) & 0xFFFFFFFF
            child_text = f"<step branch={branch_index} h={h}>"
    else:
        flat = z_star.detach().cpu().reshape(-1)[:16]
        h = int(torch.sum(flat * 1e4).item()) & 0xFFFFFFFF
        child_text = f"<step branch={branch_index} h={h}>"

    return TransitionResult(
        child_text=child_text,
        z_star_child=z_star,
        solver_stats=solver_stats,
        prune=False,
        budget=budget,
        faiss_retrieved=faiss_retrieved,
    )


def transition_batch(
    parent_text: str,
    nu: NuVector,
    budget: RuntimeBudgetState,
    backbone: BaseCTSBackbone,
    *,
    W: int = 3,
    K: int = 64,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e14,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
    faiss_context: Optional[LatentContextWindow] = None,
    fp32_buffer: bool = True,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    parent_z_star: Optional[torch.Tensor] = None,
    noise_sigma: float = 0.02,
) -> List[TransitionResult]:
    """Paper §4.1: parallel batch DEQ for W sibling branches.

    Algorithm 1 aligned: uses parent_z_star for initialization and fallback.
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d_actual = d or backbone.hidden_size

    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)

    if faiss_context is not None:
        query_z = parent_z_star if parent_z_star is not None else init_z0(K, d_actual, device, torch.Generator(device=device))
        faiss_retrieved_raw = faiss_context.retrieve(query_z, k=3)
        if faiss_retrieved_raw is not None:
            context = prepend_soft_prefix(context, faiss_retrieved_raw)

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d_actual, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()
    tol = map_nu_tol_to_tol(nu.nu_tol, broyden_tol_min, broyden_tol_max)

    z0_list = []
    for bi in range(W):
        gen = torch.Generator(device=device)
        gen.manual_seed(2026 + bi * 31 + (len(parent_text) % 997))
        if parent_z_star is not None:
            pz = parent_z_star.detach().to(device=device, dtype=torch.float32)
            if pz.shape[0] != K or pz.shape[-1] != d_actual:
                pz = init_z0(K, d_actual, device, gen)
            epsilon = torch.randn(K, d_actual, device=device, dtype=torch.float32, generator=gen) * noise_sigma * nu.nu_expl
            z0_list.append(pz + epsilon)
        else:
            z0 = init_z0(K, d_actual, device, gen)
            z0 = add_exploration_noise(z0, nu.nu_expl, gen)
            z0_list.append(z0)

    def phi(zz: torch.Tensor) -> torch.Tensor:
        if routing_mode == "dense":
            mw = routing_weights(zz, w_g, nu.nu_temp)
        else:
            mw = _routing_sparse(zz, w_g, nu.nu_temp, top_k)
        return backbone.deq_step(zz, context, mw, {"top_k": top_k})

    z0_batch = torch.stack(z0_list, dim=0)
    z_star_batch, infos = broyden_fixed_point_batch(
        phi, z0_batch, tol=tol, max_iter=broyden_max_iter,
        fp32_buffer=fp32_buffer, parent_inv_jacobian=parent_inv_jacobian,
    )

    results: List[TransitionResult] = []
    for bi, (z_star, info) in enumerate(zip(z_star_batch, infos)):
        b = budget.clone()
        b.terminal_depth += 1
        flops = 0.0
        with torch.no_grad():
            alpha = routing_weights(z_star, w_g, nu.nu_temp)
            mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
            for i in range(19):
                flops += float(mw[i].item()) * macs[i] * nu.nu_act
        b.flops_spent_step = flops
        b.mac_accumulated += flops

        solver_stats = {
            "iterations": info.iterations,
            "residual_norm": info.residual_norm,
            "converged": info.converged,
            "flops_used": flops,
        }

        if not info.converged:
            fallback_z = parent_z_star.detach().clone().to(device) if parent_z_star is not None else z_star
            results.append(TransitionResult(
                child_text=None, z_star_child=fallback_z, solver_stats=solver_stats,
                prune=True, budget=b,
            ))
            continue

        if b.mac_accumulated > tau_flops_budget * nu.nu_act:
            solver_stats["act_halt"] = True

        if faiss_context is not None:
            faiss_context.add(z_star)

        if hasattr(backbone, "decode_from_z_star"):
            try:
                import inspect as _insp
                sig = _insp.signature(backbone.decode_from_z_star)
                if "max_new_tokens" in sig.parameters:
                    child_text = backbone.decode_from_z_star(z_star, max_new_tokens=max_decode_tokens)
                else:
                    child_text = backbone.decode_from_z_star(z_star)
            except Exception:
                child_text = f"<step branch={bi}>"
        else:
            child_text = f"<step branch={bi}>"

        results.append(TransitionResult(
            child_text=child_text, z_star_child=z_star, solver_stats=solver_stats,
            prune=False, budget=b,
        ))

    return results
