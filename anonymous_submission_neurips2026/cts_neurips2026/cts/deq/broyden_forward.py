"""L-Broyden solver: limited-memory FP32 inverse Jacobian (paper §5.2).

Paper Remark 2: Inherited Jacobians warm-start non-root nodes,
yielding average 11.2+-2.8 iterations (root: 14.8; non-root: 8.9).

For small n (< MAX_DENSE_N), uses dense n×n Jacobian (fast, exact).
For large n, uses Anderson Acceleration (memory-efficient, O(m*n)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

MAX_DENSE_N = 8192


@dataclass
class BroydenInfo:
    iterations: int
    residual_norm: float
    converged: bool
    all_residuals: List[float] = field(default_factory=list)
    jacobian_state: Optional[torch.Tensor] = None


@dataclass
class BroydenConvergenceStats:
    """Aggregate statistics matching paper Appendix C / Table 12."""

    total_solves: int = 0
    converged_count: int = 0
    fallback_count: int = 0
    iteration_counts: List[int] = field(default_factory=list)
    root_iterations: List[int] = field(default_factory=list)
    nonroot_iterations: List[int] = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        return self.converged_count / max(self.total_solves, 1)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / max(self.total_solves, 1)

    @property
    def avg_iterations(self) -> float:
        return sum(self.iteration_counts) / max(len(self.iteration_counts), 1)

    @property
    def avg_root_iterations(self) -> float:
        return sum(self.root_iterations) / max(len(self.root_iterations), 1)

    @property
    def avg_nonroot_iterations(self) -> float:
        return sum(self.nonroot_iterations) / max(len(self.nonroot_iterations), 1)

    def update(self, info: BroydenInfo, is_root: bool = True) -> None:
        self.total_solves += 1
        self.iteration_counts.append(info.iterations)
        if is_root:
            self.root_iterations.append(info.iterations)
        else:
            self.nonroot_iterations.append(info.iterations)
        if info.converged:
            self.converged_count += 1
        else:
            self.fallback_count += 1

    def report(self) -> Dict[str, float]:
        return {
            "convergence_rate": self.convergence_rate,
            "fallback_rate": self.fallback_rate,
            "avg_iterations": self.avg_iterations,
            "avg_root_iterations": self.avg_root_iterations,
            "avg_nonroot_iterations": self.avg_nonroot_iterations,
            "total_solves": float(self.total_solves),
        }


_global_stats: Optional[BroydenConvergenceStats] = None


def enable_convergence_tracking() -> BroydenConvergenceStats:
    global _global_stats
    _global_stats = BroydenConvergenceStats()
    return _global_stats


def get_convergence_stats() -> Optional[BroydenConvergenceStats]:
    return _global_stats


def map_nu_tol_to_tol(nu_tol: float, tol_min: float, tol_max: float) -> float:
    """Map nu_tol in [0,1] to tolerance (monotone). Paper §4.2: [10^-4, 10^-2]."""
    n = max(0.0, min(1.0, float(nu_tol)))
    return tol_min + (tol_max - tol_min) * n


def map_nu_ne_to_tol(nu_ne: float, tol_min: float, tol_max: float) -> float:
    return map_nu_tol_to_tol(nu_ne, tol_min, tol_max)


def _dense_broyden(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int,
    parent_inv_jacobian: Optional[torch.Tensor],
    fp32_buffer: bool,
    memory_limit: int,
    is_root: bool,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """Dense Broyden for small n. Proven stable."""
    orig_shape = z0.shape
    device = z0.device
    compute_dtype = torch.float32 if fp32_buffer else z0.dtype
    z = z0.detach().reshape(-1).to(device=device, dtype=compute_dtype).clone()
    n = z.numel()

    def F(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.to(z0.dtype).view(orig_shape)
        p = phi(zz).reshape(-1).to(compute_dtype)
        return vec - p

    Fv = F(z)
    residuals: List[float] = []

    if parent_inv_jacobian is not None and parent_inv_jacobian.shape == (n, n):
        B = parent_inv_jacobian.to(device=device, dtype=compute_dtype).clone()
    else:
        B = torch.eye(n, device=device, dtype=compute_dtype)

    update_s: List[torch.Tensor] = []
    update_y: List[torch.Tensor] = []

    for it in range(max_iter):
        res = float(Fv.norm().item())
        residuals.append(res)
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            info = BroydenInfo(it + 1, res, True, residuals, jacobian_state=B.detach().clone())
            if _global_stats is not None:
                _global_stats.update(info, is_root=is_root)
            return z_out, info

        try:
            step = torch.linalg.solve(B, -Fv)
        except RuntimeError:
            step = -Fv

        z_new = z + step
        F_new = F(z_new)
        s = z_new - z
        y = F_new - Fv
        denom = torch.dot(s, s).clamp_min(1e-12)

        if len(update_s) >= memory_limit:
            update_s.pop(0)
            update_y.pop(0)
        update_s.append(s.clone())
        update_y.append(y.clone())

        B = B + torch.outer(y - B @ s, s) / denom
        z, Fv = z_new, F_new

    z_out = z.view(orig_shape).to(z0.dtype)
    info = BroydenInfo(max_iter, float(Fv.norm().item()), False, residuals, jacobian_state=B.detach().clone())
    if _global_stats is not None:
        _global_stats.update(info, is_root=is_root)
    return z_out, info


def _anderson_broyden(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int,
    fp32_buffer: bool,
    memory_limit: int,
    is_root: bool,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """Anderson acceleration for large n. O(m*n) memory."""
    orig_shape = z0.shape
    device = z0.device
    compute_dtype = torch.float32 if fp32_buffer else z0.dtype
    z = z0.detach().reshape(-1).to(device=device, dtype=compute_dtype).clone()

    def G(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.to(z0.dtype).view(orig_shape)
        p = phi(zz).reshape(-1).to(compute_dtype)
        return p

    residuals: List[float] = []
    x_hist: List[torch.Tensor] = []
    f_hist: List[torch.Tensor] = []
    beta = 1.0

    for it in range(max_iter):
        fz = G(z)
        res_vec = fz - z
        res = float(res_vec.norm().item())
        residuals.append(res)
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            info = BroydenInfo(it + 1, res, True, residuals)
            if _global_stats is not None:
                _global_stats.update(info, is_root=is_root)
            return z_out, info

        x_hist.append(z.clone())
        f_hist.append(fz.clone())
        if len(x_hist) > memory_limit + 1:
            x_hist.pop(0)
            f_hist.pop(0)

        m = len(f_hist)
        if m < 2:
            z = beta * fz + (1 - beta) * z
            continue

        dF = torch.stack([f_hist[i] - f_hist[i - 1] for i in range(1, m)], dim=0)  # [m-1, n]
        dX = torch.stack([x_hist[i] - x_hist[i - 1] for i in range(1, m)], dim=0)  # [m-1, n]

        gram = dF @ dF.t()
        gram += 1e-6 * torch.eye(m - 1, device=device, dtype=compute_dtype)
        rhs = dF @ res_vec

        try:
            gamma = torch.linalg.solve(gram, rhs)  # [m-1], small solve
        except RuntimeError:
            z = beta * fz + (1 - beta) * z
            continue

        z_new = fz - (gamma @ (dF - dX))
        z = z_new

    z_out = z.view(orig_shape).to(z0.dtype)
    fz = G(z.reshape(-1).to(compute_dtype))
    final_res = float((fz - z.reshape(-1).to(compute_dtype)).norm().item())
    info = BroydenInfo(max_iter, final_res, False, residuals)
    if _global_stats is not None:
        _global_stats.update(info, is_root=is_root)
    return z_out, info


def broyden_fixed_point(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 16,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """L-Broyden fixed-point solver (paper §5.2).

    Automatically selects dense (small n) or Anderson (large n) mode.
    Dense: O(n^2) memory, uses full Jacobian, proven stable.
    Anderson: O(m*n) memory, uses history of m steps, scalable.

    Paper rank = 16 (memory_limit default).
    """
    n = z0.numel()
    is_root = parent_inv_jacobian is None

    if n <= MAX_DENSE_N:
        return _dense_broyden(
            phi, z0, tol, max_iter, parent_inv_jacobian,
            fp32_buffer, memory_limit, is_root,
        )
    else:
        return _anderson_broyden(
            phi, z0, tol, max_iter, fp32_buffer, memory_limit, is_root,
        )


def broyden_fixed_point_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0_batch: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 16,
) -> Tuple[torch.Tensor, List[BroydenInfo]]:
    """Batch Broyden for W sibling branches (paper §4.1).

    Each branch inherits the same parent_inv_jacobian.
    """
    W = z0_batch.shape[0]
    results = []
    infos = []
    for i in range(W):
        z_star_i, info_i = broyden_fixed_point(
            phi,
            z0_batch[i],
            tol=tol,
            max_iter=max_iter,
            parent_inv_jacobian=parent_inv_jacobian,
            fp32_buffer=fp32_buffer,
            memory_limit=memory_limit,
        )
        results.append(z_star_i)
        infos.append(info_i)
    return torch.stack(results), infos
