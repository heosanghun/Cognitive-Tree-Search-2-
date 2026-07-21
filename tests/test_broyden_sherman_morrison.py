"""Regression pin: Sherman-Morrison inverse-Jacobian dense Broyden (2026-07)
is iterate-equivalent to the classic solve-against-B formulation it replaced.

The optimized `_dense_broyden` maintains H = B^-1 directly:

    step = -H @ F(z)                         ==  solve(B, -F(z))
    H'   = H - (H y - s)(s^T H) / (s^T H y)  ==  (B + (y - B s) s^T / s^T s)^-1

In exact arithmetic the iterates are identical to the historical
implementation. These tests re-implement the historical algorithm inline and
assert (a) identical root-solve iteration counts, (b) solutions agreeing well
below the solve tolerance, and (c) warm-started (inherited-Jacobian) child
solves staying within one iteration and the same tolerance band — so any
future change that silently breaks the equivalence fails loudly here.
"""

from __future__ import annotations

import torch

from cts.deq.broyden_forward import broyden_fixed_point


def _classic_dense_broyden(phi, z0, tol, max_iter, parent_B=None):
    """The pre-2026-07 formulation: store B, linalg.solve each iteration."""
    orig_shape = z0.shape
    z = z0.detach().reshape(-1).float().clone()
    n = z.numel()

    def F(v):
        return v - phi(v.view(orig_shape)).reshape(-1).float()

    Fv = F(z)
    B = parent_B.clone() if parent_B is not None else torch.eye(n)
    for it in range(max_iter):
        res = float(Fv.norm())
        if res < tol:
            return z.view(orig_shape), it + 1, res, True, B
        try:
            step = torch.linalg.solve(B, -Fv)
        except RuntimeError:
            step = -Fv
        z_new = z + step
        F_new = F(z_new)
        s, y = z_new - z, F_new - Fv
        denom = torch.dot(s, s).clamp_min(1e-12)
        B = B + torch.outer(y - B @ s, s) / denom
        z, Fv = z_new, F_new
    return z.view(orig_shape), max_iter, float(Fv.norm()), False, B


def _make_contraction(seed: int, d: int = 8):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g) * 0.05
    b = torch.randn(d, generator=g)

    def phi(zz):
        return torch.tanh(zz @ A + b) * 0.5 + zz * 0.4

    return phi


def test_sherman_morrison_matches_classic_root_solve():
    tol = 1e-6
    for seed in range(5):
        phi = _make_contraction(seed)
        z0 = torch.randn(8, 8, generator=torch.Generator().manual_seed(100 + seed))

        z_old, it_old, _, conv_old, _ = _classic_dense_broyden(phi, z0, tol, 60)
        z_new, info = broyden_fixed_point(phi, z0, tol=tol, max_iter=60)

        assert conv_old and info.converged
        assert info.iterations == it_old, (
            f"seed {seed}: SM path took {info.iterations} iterations, "
            f"classic took {it_old} — iterate equivalence broken"
        )
        assert (z_old - z_new).abs().max().item() < tol, (
            f"seed {seed}: solutions diverge beyond tol"
        )


def test_sherman_morrison_matches_classic_inherited_solve():
    """Warm-started child solves (paper Remark 2 inheritance) must agree to
    within one iteration; float round-off in the rank-1 updates makes exact
    parity too strict, but the fixed point itself must match below tol."""
    tol = 1e-6
    for seed in range(5):
        phi = _make_contraction(seed)
        z0 = torch.randn(8, 8, generator=torch.Generator().manual_seed(200 + seed))

        z_old, _, _, _, B = _classic_dense_broyden(phi, z0, tol, 60)
        _, info_root = broyden_fixed_point(phi, z0, tol=tol, max_iter=60)

        noise = torch.randn(8, 8, generator=torch.Generator().manual_seed(300 + seed))
        z0_child = z_old + 0.01 * noise

        _, it_old_c, _, conv_old_c, _ = _classic_dense_broyden(
            phi, z0_child, tol, 60, parent_B=B
        )
        z_new_c, info_c = broyden_fixed_point(
            phi, z0_child, tol=tol, max_iter=60,
            parent_inv_jacobian=info_root.jacobian_state,
        )

        assert conv_old_c and info_c.converged
        assert abs(info_c.iterations - it_old_c) <= 1, (
            f"seed {seed}: inherited solve iterations drifted "
            f"({info_c.iterations} vs {it_old_c})"
        )
        z_old_c, *_ = _classic_dense_broyden(phi, z0_child, tol, 60, parent_B=B)
        assert (z_old_c - z_new_c).abs().max().item() < tol


def test_inherited_jacobian_converges_within_cold_start_band():
    """Warm-started (inherited-H) solves must remain convergent and stay in
    the same iteration band as cold starts.

    Scope note (honest measurement, 2026-07): on these tiny synthetic
    contractions the inherited H consistently costs ~1 extra iteration versus
    a cold identity start (e.g. warm mean 8.5 vs cold 7.4 over 8 seeds) —
    and the pre-optimization solve-against-B implementation behaves the same
    way, so this is a property of good-Broyden inheritance near an already
    converged point, not of the Sherman-Morrison rewrite. Paper Remark 2's
    average-iteration reduction (14.8 root -> 8.9 non-root) is a claim about
    the real DEQ workload, exercised by the episode integration tests; this
    test therefore pins only convergence + a bounded band, not superiority.
    """
    tol = 1e-6
    cold_iters, warm_iters = [], []
    for seed in range(8):
        phi = _make_contraction(400 + seed)
        z0 = torch.randn(8, 8, generator=torch.Generator().manual_seed(500 + seed))
        z_root, info_root = broyden_fixed_point(phi, z0, tol=tol, max_iter=60)
        assert info_root.converged

        noise = torch.randn(8, 8, generator=torch.Generator().manual_seed(600 + seed))
        z0_child = z_root + 0.01 * noise

        _, info_cold = broyden_fixed_point(phi, z0_child, tol=tol, max_iter=60)
        _, info_warm = broyden_fixed_point(
            phi, z0_child, tol=tol, max_iter=60,
            parent_inv_jacobian=info_root.jacobian_state,
        )
        assert info_cold.converged and info_warm.converged
        cold_iters.append(info_cold.iterations)
        warm_iters.append(info_warm.iterations)

    mean_cold = sum(cold_iters) / len(cold_iters)
    mean_warm = sum(warm_iters) / len(warm_iters)
    assert mean_warm <= mean_cold + 2.0, (
        f"inherited-H solves left the cold-start iteration band: "
        f"warm {mean_warm:.2f} vs cold {mean_cold:.2f} ({warm_iters} vs {cold_iters})"
    )
