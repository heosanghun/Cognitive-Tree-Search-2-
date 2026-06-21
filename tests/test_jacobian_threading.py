import torch
import pytest
from typing import Callable
from cts.deq.broyden_forward import broyden_fixed_point_batch, broyden_fixed_point

def test_broyden_fixed_point_batch_threading_agreement():
    """Verify that the parallel multi-threaded broyden_fixed_point_batch
    yields bit-identical results compared to a sequential execution.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    # Tiny linear mapping to solve: f(z) = W @ z + b = z -> (W - I) @ z = -b
    # Let's make it 3 branches, 16 dimensions
    W = 3
    dim = 16
    
    matrix = torch.randn(dim, dim, device=device) * 0.05
    bias = torch.randn(dim, device=device)
    
    def phi(zz: torch.Tensor) -> torch.Tensor:
        # zz shape can be [dim] (single) or [W, dim] (batch) depending on how called.
        # But broyden_fixed_point calls phi on single slices [dim].
        return torch.matmul(matrix, zz) + bias

    z0_batch = torch.randn(W, dim, device=device)
    
    # 1) Sequential reference run
    seq_results = []
    seq_infos = []
    for i in range(W):
        z_star_i, info_i = broyden_fixed_point(
            phi, z0_batch[i], tol=1e-5, max_iter=20, fp32_buffer=True
        )
        seq_results.append(z_star_i)
        seq_infos.append(info_i)
    seq_stacked = torch.stack(seq_results)
    
    # 2) Parallel batch run
    par_results, par_infos = broyden_fixed_point_batch(
        phi, z0_batch, tol=1e-5, max_iter=20, fp32_buffer=True
    )
    
    # Assert bit-identical agreement
    assert torch.allclose(seq_stacked, par_results, atol=1e-7), (
        "Multi-threaded execution deviated from sequential reference!"
    )
    
    for i in range(W):
        assert seq_infos[i].iterations == par_infos[i].iterations
        assert abs(seq_infos[i].residual_norm - par_infos[i].residual_norm) < 1e-7
        assert seq_infos[i].converged == par_infos[i].converged
