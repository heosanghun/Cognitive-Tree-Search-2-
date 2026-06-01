import torch

from cts.deq.broyden_forward import broyden_fixed_point


def test_broyden_linear_contraction():
    def phi(z: torch.Tensor) -> torch.Tensor:
        return 0.5 * z

    z0 = torch.randn(4, 5)
    z_star, info = broyden_fixed_point(phi, z0, tol=1e-5, max_iter=40)
    assert info.converged
    assert z_star.abs().max().item() < 1e-3
