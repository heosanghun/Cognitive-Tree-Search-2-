"""Tests for L-Broyden + convergence tracking (paper §5.2, Appendix C)."""

import torch

from cts.deq.broyden_forward import (
    BroydenConvergenceStats,
    BroydenInfo,
    broyden_fixed_point,
    broyden_fixed_point_batch,
    enable_convergence_tracking,
    get_convergence_stats,
    map_nu_tol_to_tol,
    map_nu_ne_to_tol,
)


def test_map_nu_tol_monotone():
    tol_low = map_nu_tol_to_tol(0.0, 1e-4, 1e-2)
    tol_mid = map_nu_tol_to_tol(0.5, 1e-4, 1e-2)
    tol_high = map_nu_tol_to_tol(1.0, 1e-4, 1e-2)
    assert tol_low < tol_mid < tol_high


def test_legacy_alias():
    assert map_nu_ne_to_tol(0.5, 1e-4, 1e-2) == map_nu_tol_to_tol(0.5, 1e-4, 1e-2)


def test_broyden_converges_contractive():
    """A contractive map should converge."""
    def phi(z):
        return 0.5 * z + 0.1
    z0 = torch.randn(16)
    z_star, info = broyden_fixed_point(phi, z0, tol=1e-6, max_iter=50)
    assert info.converged
    assert info.iterations < 50


def test_broyden_fp32_buffer():
    """FP32 buffer should still converge."""
    def phi(z):
        return 0.3 * z + 0.2
    z0 = torch.randn(16)
    z_star, info = broyden_fixed_point(phi, z0, tol=1e-6, max_iter=50, fp32_buffer=True)
    assert info.converged


def test_broyden_records_residuals():
    def phi(z):
        return 0.5 * z
    z0 = torch.ones(8)
    _, info = broyden_fixed_point(phi, z0, tol=1e-6, max_iter=30)
    assert len(info.all_residuals) > 0
    assert info.all_residuals[-1] < info.all_residuals[0]


def test_broyden_batch():
    def phi(z):
        return 0.5 * z + 0.1
    z0_batch = torch.randn(3, 4, 8)
    z_star_batch, infos = broyden_fixed_point_batch(phi, z0_batch, tol=1e-5, max_iter=30)
    assert z_star_batch.shape == (3, 4, 8)
    assert len(infos) == 3
    assert all(i.converged for i in infos)


def test_convergence_stats():
    stats = enable_convergence_tracking()
    def phi(z):
        return 0.5 * z
    for _ in range(5):
        broyden_fixed_point(phi, torch.randn(8), tol=1e-6, max_iter=30)
    s = get_convergence_stats()
    assert s is not None
    assert s.total_solves == 5
    assert s.convergence_rate == 1.0
    report = s.report()
    assert report["convergence_rate"] == 1.0


def test_convergence_stats_report():
    stats = BroydenConvergenceStats()
    stats.update(BroydenInfo(5, 1e-7, True, []))
    stats.update(BroydenInfo(30, 0.1, False, []))
    assert stats.total_solves == 2
    assert stats.convergence_rate == 0.5
    assert stats.fallback_rate == 0.5
