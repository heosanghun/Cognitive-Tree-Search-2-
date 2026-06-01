"""Abstract backbone for CTS (swap Gemma / Llama / mock)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class BaseCTSBackbone(ABC):
    """Minimal interface for DEQ inner map f(z, context)."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @abstractmethod
    def encode_context(self, parent_text: str) -> torch.Tensor:
        """Anchored context embeddings (RoPE applied once in real model)."""

    @abstractmethod
    def deq_step(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        module_weights: torch.Tensor,
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        """One application of inner map f_theta,nu toward fixed point."""
