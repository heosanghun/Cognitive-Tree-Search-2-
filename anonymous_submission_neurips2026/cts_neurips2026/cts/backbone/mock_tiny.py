"""Tiny differentiable backbone for CPU tests (no Gemma weights)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone


class MockTinyBackbone(BaseCTSBackbone, nn.Module):
    def __init__(self, hidden: int = 64, num_layers: int = 42) -> None:
        super().__init__()
        self._hidden = hidden
        self._num_layers = num_layers
        self.embed = nn.Embedding(256, hidden)
        self.proj_z = nn.Linear(hidden, hidden)
        self.mix = nn.Linear(hidden * 2, hidden)
        self.act = nn.Tanh()

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def encode_context(self, parent_text: str) -> torch.Tensor:
        dev = self.embed.weight.device
        if not parent_text:
            ids = torch.zeros(1, dtype=torch.long, device=dev)
        else:
            ids = torch.tensor(
                [ord(c) % 256 for c in parent_text[:128]],
                dtype=torch.long,
                device=dev,
            )
        e = self.embed(ids).mean(dim=0, keepdim=True)
        return e

    def deq_step(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        module_weights: torch.Tensor,
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        # z: [K, d], context: [S, d] where S >= 1 (may include FAISS prefix)
        k, d = z.shape
        if context.shape[0] > 1:
            ctx = context.mean(dim=0, keepdim=True).expand(k, -1)
        else:
            ctx = context.expand(k, -1)
        h = torch.cat([z, ctx], dim=-1)
        delta = self.mix(h)
        out = self.act(self.proj_z(z) + delta) * 0.5 + z * 0.5
        gate = module_weights.sum() / max(module_weights.numel(), 1)
        return out * (0.9 + 0.1 * gate)
