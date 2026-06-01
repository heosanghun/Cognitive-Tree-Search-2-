"""RoPE anchor vs inner-z contract (documentation + stable API surface)."""

from cts.backbone.rope_contract import phase2_custom_forward_available, rope_policy_summary


def test_rope_policy_summary_non_empty():
    s = rope_policy_summary()
    assert "encode_context" in s or "Anchor" in s
    assert len(s) > 10


def test_phase2_not_enabled_by_default():
    assert phase2_custom_forward_available() is False
