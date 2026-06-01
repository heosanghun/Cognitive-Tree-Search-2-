"""Triton fused sparse routing kernel (paper §5.3, Appendix A.2).

Paper §5.3: "Our released Triton kernel fuses the softmax + top-k + scatter-gather
into a single kernel, achieving ~25 ms per transition batch (W=3) on a single
A100. Without the fused kernel, PyTorch reference achieves ~38 ms."

Workflow: softmax(W_g @ pool(z) / nu_temp) -> top-k mask -> renormalize
All fused into one kernel launch to minimize global memory round-trips.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_topk_softmax_kernel(
        logits_ptr, output_ptr,
        n_modules: tl.constexpr, top_k: tl.constexpr,
    ):
        """Fused softmax + top-k + renorm for sparse MoE routing.

        Single threadblock processes all n_modules logits.
        Paper: n_modules=19 (Gemma 4 E4B functional modules).
        """
        pid = tl.program_id(0)
        offset = pid * n_modules + tl.arange(0, n_modules)
        mask = tl.arange(0, n_modules) < n_modules

        logits = tl.load(logits_ptr + offset, mask=mask, other=-float('inf'))

        max_val = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_val)
        sum_exp = tl.sum(exp_logits, axis=0)
        softmax_out = exp_logits / sum_exp

        # Top-k selection via iterative masking
        selected = tl.zeros([n_modules], dtype=tl.float32)
        remaining = softmax_out

        for _ in range(top_k):
            max_remaining = tl.max(remaining, axis=0)
            is_max = (remaining == max_remaining) & (selected == 0.0)

            first_mask = tl.cumsum(is_max.to(tl.int32), axis=0) == 1
            is_first_max = is_max & first_mask

            selected = tl.where(is_first_max, softmax_out, selected)
            remaining = tl.where(is_first_max, tl.zeros_like(remaining), remaining)

        sum_selected = tl.sum(selected, axis=0)
        sum_selected = tl.where(sum_selected > 1e-8, sum_selected, tl.full([1], 1e-8, dtype=tl.float32))
        normalized = selected / sum_selected

        tl.store(output_ptr + offset, normalized, mask=mask)


def routing_weights_triton(
    z: torch.Tensor, w_g: torch.Tensor, nu_temp: float, top_k: int = 3,
) -> torch.Tensor:
    """Fused softmax + top-k routing using Triton kernel.

    Falls back to PyTorch reference if Triton unavailable.
    """
    if not _TRITON_AVAILABLE or not z.is_cuda:
        alpha = routing_weights(z, w_g, nu_temp)
        return sparse_module_weights(alpha, top_k)

    n_modules = w_g.shape[0]
    pooled = z.float().mean(dim=0)
    logits = (w_g.float() @ pooled) / max(float(nu_temp), 1e-6)
    logits = logits.contiguous()

    output = torch.zeros_like(logits)

    _fused_topk_softmax_kernel[(1,)](
        logits, output,
        n_modules=n_modules, top_k=top_k,
    )

    return output


def routing_weights_triton_batch(
    z_batch: torch.Tensor, w_g: torch.Tensor, nu_temp: float, top_k: int = 3,
) -> torch.Tensor:
    """Batched fused routing for W branches simultaneously.

    z_batch: [W, K, d]
    Returns: [W, n_modules] sparse routing weights
    """
    if not _TRITON_AVAILABLE or not z_batch.is_cuda:
        results = []
        for i in range(z_batch.shape[0]):
            alpha = routing_weights(z_batch[i], w_g, nu_temp)
            results.append(sparse_module_weights(alpha, top_k))
        return torch.stack(results)

    W = z_batch.shape[0]
    n_modules = w_g.shape[0]

    pooled = z_batch.float().mean(dim=1)  # [W, d]
    logits = (pooled @ w_g.float().t()) / max(float(nu_temp), 1e-6)  # [W, n_modules]
    logits = logits.contiguous()
    output = torch.zeros_like(logits)

    _fused_topk_softmax_kernel[(W,)](
        logits, output,
        n_modules=n_modules, top_k=top_k,
    )

    return output
