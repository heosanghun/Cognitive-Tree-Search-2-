"""
Analytic KV-cache retention model for MCTS-with-KV baseline (Table 1 contrast vs CTS).

This does **not** run a transformer; it estimates peak KV bytes from depth × sequence
growth so profiling scripts can emit comparable rows without OOM on large depths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVRetentionConfig:
    """Defaults aligned with Gemma 4 E4B text config (see `config.json` text_config)."""

    num_hidden_layers: int = 42
    num_key_value_heads: int = 2
    head_dim: int = 256
    bytes_per_element: int = 2  # bf16
    """Tokens retained along the deepest search path per tree-depth unit (tunable)."""
    tokens_per_depth_step: int = 256

    def bytes_per_layer_per_token(self) -> float:
        # K and V: 2 * num_kv_heads * head_dim * bytes
        per = self.num_key_value_heads * self.head_dim * self.bytes_per_element
        return float(2 * per)

    def estimated_peak_kv_bytes(self, tree_depth: int) -> float:
        """Linear-in-depth KV (paper-style contrast: KV MCTS blows up vs flat CTS)."""
        seq = max(1, tree_depth) * self.tokens_per_depth_step
        per_tok = self.bytes_per_layer_per_token() * float(self.num_hidden_layers)
        return per_tok * seq


def estimate_mcts_kv_peak_gb(tree_depth: int, cfg: KVRetentionConfig | None = None) -> float:
    cfg = cfg or KVRetentionConfig()
    return cfg.estimated_peak_kv_bytes(tree_depth) / 1e9


def describe_baseline() -> str:
    return (
        "Analytic KV retention: peak_gb ≈ depth × tokens_per_depth × 42 layers × "
        "2×KV heads×head_dim×2 bytes (see KVRetentionConfig)."
    )
