"""Qwen 2.5 7B backbone for CTS (paper §4.3, §5, §6)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.model.module_partition import qwen_layers_for_module


def parallel_sparse_qwen_module_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: torch.Tensor,
    module_alpha: torch.Tensor,
    layers_for_module_fn,
    top_k: int = 3,
) -> torch.Tensor:
    """
    Sum_m alpha_m * Module_m(z) for Qwen (no PLE or anchor pre-population needed).
    """
    device = z.device
    model_dtype = next(lm.parameters()).dtype
    z = z.to(dtype=model_dtype)
    dtype = model_dtype
    k, h = z.shape
    ctx = context_row.to(device=device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    h0 = (z + ctx.expand(k, -1)).unsqueeze(0)

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

    position_embeddings = lm.rotary_emb(h0, position_ids)

    m = module_alpha.numel()
    k_top = min(top_k, m)
    sel_alpha, top_idx = torch.topk(module_alpha, k_top)
    sel_alpha = sel_alpha / (sel_alpha.sum().clamp_min(1e-8))

    acc = torch.zeros_like(h0)
    for j, mod in enumerate(top_idx.tolist()):
        a = sel_alpha[j]
        hidden = h0.clone()
        layer_ids: List[int] = layers_for_module_fn(mod)

        for li in layer_ids:
            out = lm.layers[li](
                hidden,
                attention_mask=causal_mask_mapping[lm.config.layer_types[li]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            hidden = out[0] if isinstance(out, tuple) else out

        acc = acc + a.to(hidden.dtype) * hidden

    acc = lm.norm(acc)
    return acc.squeeze(0)


def full_stack_qwen_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: torch.Tensor,
) -> torch.Tensor:
    """
    One full sequential pass through all Qwen decoder layers.
    """
    device = z.device
    model_dtype = next(lm.parameters()).dtype
    z = z.to(dtype=model_dtype)
    dtype = model_dtype
    k, h = z.shape
    ctx = context_row.to(device=device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    h0 = (z + ctx.expand(k, -1)).unsqueeze(0)

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

    position_embeddings = lm.rotary_emb(h0, position_ids)

    hidden_states = h0
    for i, decoder_layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        out = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[lm.config.layer_types[i]],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = out[0] if isinstance(out, tuple) else out

    hidden_states = lm.norm(hidden_states)
    return hidden_states.squeeze(0)


class QwenCTSBackbone(BaseCTSBackbone, nn.Module):
    def __init__(self, cg_model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.cg = cg_model
        self.tokenizer = tokenizer
        
        cfg = cg_model.config
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        self._hidden = int(cfg.hidden_size)
        self._num_layers = int(cfg.num_hidden_layers)
        
        dev = next(cg_model.parameters()).device
        dt = next(cg_model.parameters()).dtype
        
        # Qwen has 14 modules
        self.routing_proj = nn.Parameter(torch.randn(14, self._hidden, device=dev, dtype=dt) * 0.02)
        
        self._blend = nn.Linear(self._hidden, self._hidden, bias=True).to(device=dev)
        nn.init.normal_(self._blend.weight, std=0.02)
        nn.init.zeros_(self._blend.bias)

        self.w_proj = nn.Linear(self._hidden, self._hidden, bias=False).to(device=dev, dtype=dt)
        nn.init.normal_(self.w_proj.weight, std=0.02)

        self.deq_map_mode = os.environ.get("CTS_DEQ_MAP_MODE", "blend")

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def _device(self) -> torch.device:
        return next(self.cg.parameters()).device

    def routing_matrix(self) -> torch.Tensor:
        return self.routing_proj

    def encode_context(self, parent_text: str) -> torch.Tensor:
        enc = self.tokenizer(
            parent_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = self._device()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.set_grad_enabled(self.training):
            out = self.cg.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        h = out.last_hidden_state.float()
        ctx = h.mean(dim=1)
        return ctx.to(dtype=next(self.cg.parameters()).dtype)

    def deq_step(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        module_weights: torch.Tensor,
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        mode = extra.get("deq_map_mode", self.deq_map_mode)
        top_k = int(extra.get("top_k", 3))
        ctx = context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)

        if mode in ("parallel", "paper"):
            out = parallel_sparse_qwen_module_forward(
                self.cg.model, z, ctx, module_weights, qwen_layers_for_module, top_k=top_k,
            )
            return out.to(dtype=z.dtype)

        if mode == "full":
            out = full_stack_qwen_forward(self.cg.model, z, ctx)
            return out.to(dtype=z.dtype)

        zf = z.float()
        ctxf = ctx.float().expand(zf.shape[0], -1)
        gate = float(module_weights.sum().item() / max(module_weights.numel(), 1))
        gate = max(0.25, min(1.5, gate))
        h = zf + ctxf
        delta = self._blend(h)
        mixed = 0.82 * zf + 0.18 * gate * torch.tanh(delta)
        return mixed.to(dtype=z.dtype)

    def decode_from_z_star(
        self,
        z_star: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        problem_text: Optional[str] = None,
    ) -> str:
        if max_new_tokens <= 0:
            return ""
        dt = next(self.cg.parameters()).dtype
        device = self._device()

        z_projected = self.w_proj(z_star.to(dtype=dt))
        prefix_embeds = z_projected.unsqueeze(0)  # [1, K, d_model]

        if problem_text:
            try:
                tok_out = self.tokenizer(
                    problem_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1024,
                )
                tid_text = tok_out["input_ids"].to(device)
                txt_embeds = self.cg.model.get_input_embeddings()(tid_text).to(dtype=dt)
                prefix_embeds = torch.cat([prefix_embeds, txt_embeds], dim=1)
            except Exception:
                pass
        prefix_len = prefix_embeds.shape[1]

        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            try:
                eos = int(self.cg.config.eos_token_id)
            except Exception:
                eos = 151643

        ids: List[int] = []
        try:
            with torch.no_grad():
                attn_mask = torch.ones(1, prefix_len, device=device, dtype=torch.long)
                out = self.cg.model(
                    inputs_embeds=prefix_embeds,
                    attention_mask=attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                h = out.last_hidden_state[:, -1, :]
                logits = self.cg.lm_head(h)
                next_id = int(logits.argmax(dim=-1).item())
                ids.append(next_id)

                for _ in range(max_new_tokens - 1):
                    if next_id == eos:
                        break
                    tid_t = torch.tensor([[next_id]], device=device, dtype=torch.long)
                    out = self.cg.model(
                        input_ids=tid_t,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values
                    h = out.last_hidden_state[:, -1, :]
                    logits = self.cg.lm_head(h)
                    next_id = int(logits.argmax(dim=-1).item())
                    ids.append(next_id)

            return self.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            h = z_star.mean(dim=0).to(dtype=dt)
            logits = self.cg.lm_head(h.unsqueeze(0)).squeeze(0)
            tid = int(logits.argmax(dim=-1).item())
            return self.tokenizer.decode([tid], skip_special_tokens=True)
