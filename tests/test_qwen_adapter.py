"""Unit tests for Qwen backbone adapter."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from cts.backbone.qwen_adapter import QwenCTSBackbone


class MockQwenConfig:
    def __init__(self, hidden_size: int = 32, num_hidden_layers: int = 28) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = ["full_attention"] * num_hidden_layers
        self.unique_layer_types = ["full_attention"]
        self.eos_token_id = 151643
        self._attn_implementation = "eager"

    def get_text_config(self) -> MockQwenConfig:
        return self





class MockRotaryEmb(nn.Module):
    def forward(self, h0: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn_like(h0), torch.randn_like(h0))


class MockLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return hidden_states + 0.01 * torch.tanh(hidden_states)


class MockQwenModel(nn.Module):
    def __init__(self, config: MockQwenConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([MockLayer() for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(32)
        self.rotary_emb = MockRotaryEmb()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> Any:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        class Output:
            def __init__(self, last_hidden_state: torch.Tensor, past_key_values: Any = None) -> None:
                self.last_hidden_state = last_hidden_state
                self.past_key_values = past_key_values

        # Mock cache return when use_cache is True
        past_kvs = None
        if use_cache:
            past_kvs = object()

        return Output(inputs_embeds, past_key_values=past_kvs)


class MockQwenLM(nn.Module):
    def __init__(self, config: MockQwenConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MockQwenModel(config)
        self.lm_head = nn.Linear(32, 100)


class MockTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 151643

    def __call__(self, text: str, return_tensors: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "mock dec completion"


def test_qwen_adapter_cpu():
    cfg = MockQwenConfig()
    model = MockQwenLM(cfg)
    tok = MockTokenizer()

    bb = QwenCTSBackbone(model, tok)
    assert bb.num_layers == 28
    assert bb.hidden_size == 32

    # 1. Test encode_context
    ctx = bb.encode_context("hello world")
    assert ctx.shape == (1, 32)

    # 2. Test deq_step in full mode
    z = torch.randn(4, 32)
    out_full = bb.deq_step(z, ctx, torch.ones(14), {"deq_map_mode": "full"})
    assert out_full.shape == (4, 32)

    # 3. Test deq_step in parallel mode
    out_parallel = bb.deq_step(z, ctx, torch.ones(14), {"deq_map_mode": "parallel", "top_k": 3})
    assert out_parallel.shape == (4, 32)

    # 4. Test deq_step in blend mode
    out_blend = bb.deq_step(z, ctx, torch.ones(14), {"deq_map_mode": "blend"})
    assert out_blend.shape == (4, 32)

    # 5. Test decode_from_z_star
    z_star = torch.randn(4, 32)
    ans = bb.decode_from_z_star(z_star, max_new_tokens=5, problem_text="test problem")
    assert ans == "mock dec completion"
