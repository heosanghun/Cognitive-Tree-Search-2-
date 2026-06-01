import torch

from cts.train.routing_proj_step import MockRoutingOnly, routing_entropy, routing_loss_paper_style


def test_entropy_positive():
    bb = MockRoutingOnly(d=16)
    z = torch.randn(4, 16)
    h = routing_entropy(z, bb, nu_temp=1.0)
    assert h > 0


def test_paper_style_loss_with_entropy():
    bb = MockRoutingOnly(d=16)
    z = torch.randn(4, 16)
    l = routing_loss_paper_style(z, bb, entropy_coef=0.01)
    assert l.ndim == 0
