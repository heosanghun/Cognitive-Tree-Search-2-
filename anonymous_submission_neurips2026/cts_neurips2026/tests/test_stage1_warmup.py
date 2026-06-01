import torch

from cts.train.stage1_warmup import fixed_point_surrogate_loss, run_stage1_demo_step
from cts.types import NuVector


def test_fixed_point_loss_scalar():
    loss, _ = run_stage1_demo_step(lr=1e-2)
    assert loss == loss  # finite
    assert loss >= 0.0


def test_surrogate_backward():
    from cts.backbone.mock_tiny import MockTinyBackbone

    bb = MockTinyBackbone(hidden=32, num_layers=8)
    nu = NuVector(nu_temp=1.0)
    w_g = torch.randn(19, 32) * 0.02
    z = torch.randn(4, 32)
    loss = fixed_point_surrogate_loss(bb, "x", z, nu, w_g=w_g)
    loss.backward()
    assert bb.mix.weight.grad is not None
