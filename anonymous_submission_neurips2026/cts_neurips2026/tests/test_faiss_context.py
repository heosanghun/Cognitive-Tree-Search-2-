"""Tests for FAISS Latent Space Context Window (paper §4.4)."""

import torch
import pytest

from cts.latent.faiss_context import LatentContextWindow, prepend_soft_prefix


def test_context_window_add_and_size():
    ctx = LatentContextWindow(dim=32, retrieval_k=3, min_steps=5)
    assert ctx.size == 0
    z = torch.randn(8, 32)
    ctx.add(z)
    assert ctx.size == 1
    assert ctx.step_count == 1


def test_context_window_retrieve_returns_none_below_min():
    ctx = LatentContextWindow(dim=32, retrieval_k=3, min_steps=10)
    for _ in range(9):
        ctx.add(torch.randn(8, 32))
    assert ctx.retrieve(torch.randn(8, 32)) is None


def test_context_window_retrieve_after_min_steps():
    ctx = LatentContextWindow(dim=32, retrieval_k=3, min_steps=5)
    for _ in range(10):
        ctx.add(torch.randn(8, 32))
    result = ctx.retrieve(torch.randn(8, 32))
    assert result is not None
    assert result.shape == (3, 8, 32)  # full K x d FP16 vectors (paper §4.3)


def test_context_window_memory_kb():
    ctx = LatentContextWindow(dim=64, retrieval_k=3, min_steps=0)
    for _ in range(100):
        ctx.add(torch.randn(8, 64))
    kb = ctx.memory_kb_per_node()
    assert kb > 0
    assert kb < 10.0  # full K x d FP16 (~1 KB/node for K=8, d=64)


def test_context_window_reset():
    ctx = LatentContextWindow(dim=32, retrieval_k=3, min_steps=0)
    for _ in range(5):
        ctx.add(torch.randn(8, 32))
    assert ctx.size == 5
    ctx.reset()
    assert ctx.size == 0


def test_prepend_soft_prefix():
    context = torch.randn(10, 64)
    retrieved_2d = torch.randn(3, 64)
    result_2d = prepend_soft_prefix(context, retrieved_2d)
    assert result_2d.shape == (13, 64)
    assert torch.allclose(result_2d[3:], context)

    retrieved_3d = torch.randn(3, 8, 64)  # k, K, d
    result_3d = prepend_soft_prefix(context, retrieved_3d)
    assert result_3d.shape == (24 + 10, 64)  # 3*8 + 10


def test_transition_with_faiss_context():
    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState

    bb = MockTinyBackbone(hidden=32, num_layers=42)
    ctx = LatentContextWindow(dim=32, retrieval_k=3, min_steps=2)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)

    converged_count = 0
    for i in range(5):
        r = transition(
            f"test prompt {i}", i, nu, RuntimeBudgetState(), bb,
            K=4, d=32, broyden_max_iter=40, faiss_context=ctx
        )
        if r.solver_stats["converged"]:
            converged_count += 1
    assert converged_count >= 3, "Most branches should converge"
    assert ctx.size > 0
