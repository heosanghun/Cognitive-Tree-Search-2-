"""
Gemma 4 text LM forward on continuous latent z (soft thought bottleneck).

Gemma4TextModel requires per-layer embeddings (PLE). Passing **zeros** as the
second argument to `project_per_layer_inputs` avoids discrete-token `get_per_layer_inputs`
while still producing valid [B,T,L,d_pl] tensors (see HF Gemma4TextModel).
"""

from __future__ import annotations

from typing import List

import torch


def project_ple_inputs(
    lm: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    zero_init: bool = True,
) -> torch.Tensor:
    """Returns combined per-layer inputs [B,T,num_layers,d_pl]."""
    cfg = lm.config
    b, t, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    nL = cfg.num_hidden_layers
    dpl = cfg.hidden_size_per_layer_input
    if zero_init:
        zpl = torch.zeros(b, t, nL, dpl, device=device, dtype=dtype)
    else:
        zpl = None
    return lm.project_per_layer_inputs(inputs_embeds, zpl)


def full_stack_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: torch.Tensor,
) -> torch.Tensor:
    """
    One full sequential pass through all decoder layers (paper inner map without sparse ablation).

    z: [K, H], context_row: [1, H] — context is added to each latent slot (conditioning).
    Returns z_out [K, H].
    """
    device = z.device
    # transformers 5.x Gemma 4 loads `per_layer_model_projection` and the
    # decoder layers in BF16. The Broyden FP iteration uses an fp32 buffer
    # (see broyden_forward.py:`compute_dtype`) so callers without an
    # autocast context (e.g. Stage 2 PPO rollouts under torch.no_grad)
    # otherwise hit "expected mat1 and mat2 to have the same dtype". Cast
    # z to the model's parameter dtype here; downstream tensors (ctx,
    # h0, inputs_embeds, per_layer_inputs) inherit it so the linear
    # multiplications stay homogeneous. Stage 1 was already protected by
    # autocast(bf16), but the fix is local and safe in either context.
    model_dtype = next(lm.parameters()).dtype
    z = z.to(dtype=model_dtype)
    dtype = model_dtype
    k, h = z.shape
    ctx = context_row.to(device=device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    h0 = z + ctx.expand(k, -1)
    inputs_embeds = h0.unsqueeze(0)
    attention_mask = torch.ones(1, k, device=device, dtype=torch.long)
    past_seen = 0
    position_ids = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0) + past_seen

    per_layer_inputs = project_ple_inputs(lm, inputs_embeds, zero_init=True)

    # Match HF: build causal mask dict for sliding/full attention layers
    from transformers.masking_utils import create_masks_for_generate

    causal_mask_mapping = create_masks_for_generate(
        lm.config,
        inputs_embeds,
        attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )

    position_embeddings = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(inputs_embeds, position_ids, layer_type)

    hidden_states = inputs_embeds
    # transformers 5.x Gemma 4 layers require a `shared_kv_states` dict
    # so the kv-sharing decoder layers (paper §5: 16 SWA + 3 global; the
    # last few layers reuse the kv of the last non-shared layer of the
    # same type) can read/write the shared kv. The standard model
    # forward in modeling_gemma4.py L1616 also initialises this as `{}`
    # and lets the layers populate it as they execute. Older transformers
    # (<5) did not have this argument at this layer; we therefore guard
    # the call so the same code path keeps working on both ABIs.
    shared_kv_states: dict = {}
    layer_kwargs = {
        "position_ids": position_ids,
        "past_key_values": None,
    }
    for i, decoder_layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        pli = per_layer_inputs[:, :, i, :]
        hidden_states = decoder_layer(
            hidden_states,
            pli,
            shared_kv_states=shared_kv_states,
            position_embeddings=position_embeddings[lm.config.layer_types[i]],
            attention_mask=causal_mask_mapping[lm.config.layer_types[i]],
            **layer_kwargs,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

    hidden_states = lm.norm(hidden_states)
    return hidden_states.squeeze(0)


def parallel_sparse_module_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: torch.Tensor,
    module_alpha: torch.Tensor,
    layers_for_module,
    top_k: int = 3,
) -> torch.Tensor:
    """
    Paper Eq.(5)-style: sum_m alpha_m * Module_m(z), each module runs its layer subset from shared h0.

    module_alpha: [19] softmax weights (already sparse if desired).
    """
    device = z.device
    # See `full_stack_forward` for the rationale: force the model's
    # parameter dtype here so callers outside an autocast block (e.g.
    # Stage 2 PPO rollouts under torch.no_grad) match the BF16 weights
    # of `per_layer_model_projection` and the decoder layers. We do NOT
    # cast `module_alpha` itself (gradient must flow back to
    # routing_proj in Stage 1) — instead it is cast at the multiplication
    # site below.
    model_dtype = next(lm.parameters()).dtype
    z = z.to(dtype=model_dtype)
    dtype = model_dtype
    k, h = z.shape
    ctx = context_row.to(device=device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    h0 = (z + ctx.expand(k, -1)).unsqueeze(0)
    per_layer_inputs = project_ple_inputs(lm, h0, zero_init=True)

    attention_mask = torch.ones(1, k, device=device, dtype=torch.long)
    position_ids = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0)
    from transformers.masking_utils import create_masks_for_generate

    causal_mask_mapping = create_masks_for_generate(
        lm.config,
        h0,
        attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(h0, position_ids, layer_type)

    m = module_alpha.numel()
    k_top = min(top_k, m)
    sel_alpha, top_idx = torch.topk(module_alpha, k_top)
    sel_alpha = sel_alpha / (sel_alpha.sum().clamp_min(1e-8))

    # Pre-populate `shared_kv_states` for all kv-anchor layers using
    # the shared `h0` input. transformers 5.x splits Gemma 4 layers into
    # "anchor" layers (`store_full_length_kv=True`, the last non-shared
    # layer of each attention type) that WRITE into the shared dict,
    # and downstream "kv-shared" layers (`is_kv_shared_layer=True`)
    # that READ from it. In a normal sequential model.forward the dict
    # is populated as the loop iterates; here we call only the top-k
    # module subsets, which may not include the anchor for a given
    # type. Without this pre-fill the first kv-shared layer would hit
    # `KeyError` because its `kv_shared_layer_index` was never written.
    # Running anchors against `h0` matches the paper §4.2 design: each
    # module is module-independent and starts from the shared
    # post-routing context, so the anchor's "context view" must be the
    # same `h0` every module sees. This is bit-for-bit equivalent to
    # the previous transformers (<5) behaviour where kv-sharing was
    # transparent and module forwards were stateless across modules.
    shared_kv_states: dict = {}
    anchor_indices: List[int] = []
    for li in range(lm.config.num_hidden_layers):
        attn = getattr(lm.layers[li], "self_attn", None)
        if attn is not None and getattr(attn, "store_full_length_kv", False):
            anchor_indices.append(li)
    for li in anchor_indices:
        pli_a = per_layer_inputs[:, :, li, :]
        with torch.no_grad():
            _ = lm.layers[li](
                h0,
                pli_a,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[lm.config.layer_types[li]],
                attention_mask=causal_mask_mapping[lm.config.layer_types[li]],
                position_ids=position_ids,
                past_key_values=None,
            )

    acc = torch.zeros_like(h0)
    for j, mod in enumerate(top_idx.tolist()):
        # NOTE: previously this used ``a = float(sel_alpha[j].item())``,
        # which detached ``a`` from the autograd graph and meant that
        # ``acc = acc + a * hidden`` could not propagate gradients back
        # through ``module_alpha`` (and therefore back through
        # ``routing_proj``). Keep the alpha as a tensor so the routing
        # projection receives gradient signal during Stage 1.
        a = sel_alpha[j]
        hidden = h0.clone()
        layer_ids: List[int] = layers_for_module(mod)
        # Each module reuses the pre-filled anchor kv (read-only per
        # module) but has its own writes layered on top. We copy to
        # avoid leaking the per-module writes back into other modules'
        # views.
        per_module_kv = dict(shared_kv_states)
        for li in layer_ids:
            pli = per_layer_inputs[:, :, li, :]
            out = lm.layers[li](
                hidden,
                pli,
                shared_kv_states=per_module_kv,
                position_embeddings=position_embeddings[lm.config.layer_types[li]],
                attention_mask=causal_mask_mapping[lm.config.layer_types[li]],
                position_ids=position_ids,
                past_key_values=None,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        # Cast `a` only at the multiplication site so the routing
        # projection still receives gradient through the original
        # (typically fp32) softmax distribution.
        acc = acc + a.to(hidden.dtype) * hidden

    acc = lm.norm(acc)
    return acc.squeeze(0)
