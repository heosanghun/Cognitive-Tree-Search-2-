import torch

from cts.train.routing_proj_step import MockRoutingOnly, routing_target_alignment_loss, train_routing_proj_one_step


def test_mock_routing_one_step():
    bb = MockRoutingOnly(d=32)
    z = torch.randn(4, 32)
    loss0 = routing_target_alignment_loss(z, bb, nu_temp=1.0)
    assert loss0.ndim == 0
    loss1, bb2 = train_routing_proj_one_step(bb, z=z, lr=0.05)
    assert isinstance(loss1, float)
