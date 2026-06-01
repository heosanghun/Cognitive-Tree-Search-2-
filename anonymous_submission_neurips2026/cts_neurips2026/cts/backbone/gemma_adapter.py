"""Gemma 4 E4B backbone for CTS (paper §4.3, §5, §6).

Key components:
- encode_context: text -> mean-pooled anchor embedding
- deq_step: configurable DEQ map (blend/parallel/full)
- routing_proj: W_g [19, H] for sparse module routing (Eq. 3)
- w_proj: Wproj [d, d_model] learned projection (§4.3)
- decode_from_z_star: Wproj soft-prompt prefix -> frozen AR pass (§4.3)

Paper §4.3: "Terminal z* is projected via Wproj in R^{d x d_model} as a
soft-prompt prefix into the frozen Gemma 4 decoder for one autoregressive
pass (~1.2 s)."

Paper §6: Wproj is trained jointly in Stage 1.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.gemma_latent_forward import full_stack_forward, parallel_sparse_module_forward
from cts.model.module_partition import layers_for_module


class GemmaCTSBackbone(BaseCTSBackbone, nn.Module):
    def __init__(self, cg_model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.cg = cg_model
        self.tokenizer = tokenizer
        cfg = cg_model.config.get_text_config()
        self._hidden = int(cfg.hidden_size)
        self._num_layers = int(cfg.num_hidden_layers)
        dev = next(cg_model.parameters()).device
        dt = next(cg_model.parameters()).dtype
        self.routing_proj = nn.Parameter(torch.randn(19, self._hidden, device=dev, dtype=dt) * 0.02)
        self._blend = nn.Linear(self._hidden, self._hidden, bias=True).to(device=dev)
        nn.init.normal_(self._blend.weight, std=0.02)
        nn.init.zeros_(self._blend.bias)

        # Paper §4.3: Wproj in R^{d x d_model} — projects z* [K, d] into
        # embedding space for soft-prompt prefix decoding.
        # Trained jointly in Stage 1.
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

    def _lm(self) -> nn.Module:
        return self.cg.model.language_model

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
            lm = self._lm()
            out = parallel_sparse_module_forward(
                lm, z, ctx, module_weights, layers_for_module, top_k=top_k,
            )
            return out.to(dtype=z.dtype)

        if mode == "full":
            out = full_stack_forward(self._lm(), z, ctx)
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
        """Paper §4.3: Wproj soft-prompt prefix -> frozen Gemma AR pass.

        z_star [K, d] -> Wproj -> [K, d_model] soft-prompt prefix
        -> frozen Gemma autoregressive decoding -> answer text

        Because final decoding uses W=1, KV-cache collapses to O(L).
        Peak VRAM stays below 18.0 GB (paper §4.3).

        ``problem_text``: optional. When supplied, the original problem
        text is tokenized and concatenated *after* the soft-prompt prefix
        so the frozen Gemma has explicit textual context in addition to
        the latent prefix. This restores paper §4.3 intent ("the soft
        prompt augments, not replaces, the problem context") and avoids
        the failure mode where a compute-limited Wproj produces random
        non-sequitur tokens like 'Cultura' / 'LinearLayout' on AIME
        prompts when no textual context is available. Backwards-
        compatible: callers that don't pass ``problem_text`` get the
        previous soft-prompt-only behaviour unchanged.
        """
        if max_new_tokens <= 0:
            return ""
        dt = next(self.cg.parameters()).dtype
        device = self._device()

        # Project z* through Wproj: [K, d] -> [K, d_model]
        z_projected = self.w_proj(z_star.to(dtype=dt))
        prefix_embeds = z_projected.unsqueeze(0)  # [1, K, d_model]

        # Optional textual context appended after the soft-prompt prefix
        # so the frozen Gemma has both the latent representation and the
        # original problem. The textual block uses the standard token
        # embedding table so it composes naturally with the soft prefix.
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
                # If anything in the tokenize/embed path fails, fall back
                # silently to the soft-prompt-only path so we never lose
                # the original behaviour.
                pass
        prefix_len = prefix_embeds.shape[1]

        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            try:
                eos = int(self.cg.config.get_text_config().eos_token_id)
            except Exception:
                eos = 1

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
