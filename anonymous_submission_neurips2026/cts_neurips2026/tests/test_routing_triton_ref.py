"""Triton routing path must match reference softmax + top-k renormalize."""

import torch

from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.routing.sparse_moe_triton import routing_weights_triton


def test_triton_routing_matches_ref():
    torch.manual_seed(0)
    z = torch.randn(8, 64)
    w_g = torch.randn(19, 64) * 0.02
    k = 3
    nu = 1.0
    alpha = routing_weights(z, w_g, nu)
    ref = sparse_module_weights(alpha, k)
    tri = routing_weights_triton(z, w_g, nu, k)
    assert torch.allclose(ref, tri, atol=1e-6, rtol=1e-5)
