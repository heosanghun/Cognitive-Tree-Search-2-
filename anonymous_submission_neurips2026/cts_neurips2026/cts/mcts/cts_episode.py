"""Algorithm 1: Cognitive Tree Search (CTS) — Single Episode.

Paper-aligned full MCTS episode loop with:
  - PUCT selection across the full tree (Eq. 2)
  - MetaPolicy nu sampling (§4.1)
  - W parallel DEQ transitions with parent z* noise (line 6)
  - Neuro-Critic Q evaluation (line 12)
  - FAISS registration (line 12)
  - Tree backpropagation (line 14)
  - ACT halting (line 15)
  - Best-trajectory decoding via Wproj (line 18)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cts.backbone.protocol import BaseCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.deq.transition import transition
from cts.latent.bottleneck import init_z0
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.hybrid_kv import HybridKVManager, hybrid_transition_decision
from cts.mcts.puct import PUCTVariant, select_action
from cts.mcts.tree import SearchTree
from cts.policy.meta_policy import MetaPolicy
from cts.types import (
    NuConfigMode,
    NuVector,
    RuntimeBudgetState,
    TransitionResult,
    TreeNode,
)


@dataclass
class CtsEpisodeResult:
    """Output of a single CTS episode."""
    answer: str
    best_z_star: Optional[torch.Tensor]
    tree: SearchTree
    total_mac: float
    total_iterations: int
    stats: Dict[str, Any] = field(default_factory=dict)


def _pool_z_star(z: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Mean-pool z* [K, d] -> [d] for MetaPolicy / Critic input."""
    if z is None:
        return None
    return z.detach().float().mean(dim=0)


def _backpropagate(tree: SearchTree, node_id: int, q_values: List[float]) -> None:
    """Backpropagate mean Q from children up to root (paper line 14)."""
    node = tree.nodes[node_id]
    if not q_values:
        return
    mean_q = sum(q_values) / len(q_values)

    cur_id: Optional[int] = node_id
    while cur_id is not None:
        cur = tree.nodes[cur_id]
        old_n = cur.mcts_N
        cur.mcts_N = old_n + 1
        if cur.parent_id is not None:
            parent = tree.nodes[cur.parent_id]
            child_idx = parent.children_ids.index(cur_id) if cur_id in parent.children_ids else 0
            if child_idx < len(parent.mcts_Q):
                old_q = parent.mcts_Q[child_idx]
                old_visits = max(1, old_n)
                parent.mcts_Q[child_idx] = (old_q * old_visits + mean_q) / (old_visits + 1)
        cur_id = cur.parent_id


def _select_leaf(
    tree: SearchTree,
    nu_expl: float,
    variant: PUCTVariant = "paper",
    *,
    nu_temp: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """PUCT tree traversal: select a leaf node for expansion (paper line 3).

    When ``nu_temp > 0``, Gumbel(0, nu_temp) noise is added to each child's
    PUCT score before argmax. This is the canonical way to wire the
    meta-policy's temperature output into a discrete action choice and is
    the missing link that previously caused multi-seed CTS-4nu runs to
    collapse to identical greedy outputs (std=0.0).

    ``generator`` is an optional ``torch.Generator`` used to draw the Gumbel
    samples deterministically; pass ``selection_seed`` from
    ``cts_full_episode`` to make per-(seed, problem) exploration
    reproducible.
    """
    from cts.mcts.puct import puct_score

    cur = 0
    while tree.nodes[cur].children_ids:
        node = tree.nodes[cur]
        children = node.children_ids
        n_parent = max(1, node.mcts_N)

        best_score = float("-inf")
        best_child = children[0]
        for idx, cid in enumerate(children):
            child = tree.nodes[cid]
            prior = node.mcts_prior[idx] if idx < len(node.mcts_prior) else 1.0 / len(children)
            q = node.mcts_Q[idx] if idx < len(node.mcts_Q) else 0.0
            n_sa = child.mcts_N

            score = puct_score(variant, nu_expl, prior, n_parent, n_sa, q)

            if nu_temp > 0.0:
                # Gumbel(0, nu_temp): -nu_temp * log(-log(U)) with U ~ U(0, 1)
                u = torch.rand((), generator=generator).item()
                u = min(max(u, 1e-9), 1.0 - 1e-9)
                gumbel = -nu_temp * float(torch.log(-torch.log(torch.tensor(u))))
                score = score + gumbel

            if score > best_score:
                best_score = score
                best_child = cid
        cur = best_child
    return cur


def cts_full_episode(
    prompt: str,
    *,
    backbone: BaseCTSBackbone,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    W: int = 3,
    K: int = 64,
    tau_budget: float = 1e14,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    top_k: int = 3,
    puct_variant: PUCTVariant = "paper",
    faiss_context: Optional[LatentContextWindow] = None,
    hybrid_kv_manager: Optional[HybridKVManager] = None,
    max_decode_tokens: int = 64,
    routing_mode: str = "sparse",
    noise_sigma: float = 0.02,
    device: Optional[torch.device] = None,
    wall_clock_budget_s: Optional[float] = None,
    z0_seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    nu_config_mode: Optional[NuConfigMode] = None,
    nu_trace: Optional[List[NuVector]] = None,
    k_override: Optional[int] = None,
    w_override: Optional[int] = None,
) -> CtsEpisodeResult:
    """Algorithm 1: Cognitive Tree Search — Single Episode.

    Require: Prompt s0, budget tau, W, f_theta, pi_phi, V_psi, FAISS F
    Ensure: Decoded answer y_hat

    ``nu_config_mode``: paper Table 5 nu-component Pareto switch.
        ``None`` or ``"4nu"`` keeps every meta-policy output live (CTS-4nu).
        ``"2nu_fast"`` keeps {expl, temp} active and freezes {tol, act} at
        the Stage 1 converged means; this is the canonical CTS-2nu fast
        mode used in Table 5. Other valid modes: ``"3nu_no_act"``,
        ``"2nu_expl_tol"``, ``"1nu"``. Mode switching requires NO
        retraining (paper §7.5 footnote).

    ``nu_trace``: optional caller-owned list. When provided, every NuVector
        sampled by the meta-policy (after any ``nu_config_mode`` override)
        is appended to it. This is the persistence hook used by
        ``cts/eval/nu_stats.py`` to reproduce the paper Table 19 per-domain
        ν statistics. ``None`` (the default) keeps the historical zero-overhead
        behaviour so the existing 308 tests stay green.

    ``k_override``: paper Table 13 (MCTS top-K children sensitivity sweep).
        When provided, overrides the per-leaf children-expansion count
        (otherwise equal to ``W``) without retraining. ``None`` (default)
        preserves the historical behaviour where ``W`` controls both the
        branching factor and the per-step simulation budget. NOTE: this is
        a DIFFERENT "K" than the latent-token count ``K`` (paper §4.2,
        README "Configuration" K=64); the latter is the bottleneck width
        of z*_t and is unaffected.

    ``w_override``: paper Table 15 (per-step MCTS simulation budget scaling
        sweep). When provided, caps the outer PUCT loop at exactly
        ``w_override`` iterations regardless of ``tau_budget`` / ACT halting.
        ``None`` (default) preserves the historical compute-budget halting
        behaviour. NOTE: this is a DIFFERENT "W" than the branching factor
        ``W`` (paper §4.1, README "Configuration" W=3); the latter is the
        children-per-leaf parallelism and is unaffected unless
        ``k_override`` is set.
    """
    if device is None:
        if hasattr(backbone, "parameters"):
            device = next(backbone.parameters()).device
        else:
            device = torch.device("cpu")

    d = backbone.hidden_size

    # Sweep overrides (paper Table 13 / Table 15). These default to ``None`` so
    # existing call sites are byte-identical; pinned by
    # ``tests/test_sweep_K_W_lambda.py::test_cts_full_episode_accepts_k_and_w_override``.
    W_eff = int(k_override) if k_override is not None else int(W)
    if W_eff < 1:
        raise ValueError(f"k_override must be >= 1 (got {k_override})")
    sim_cap: Optional[int] = None
    if w_override is not None:
        sim_cap = int(w_override)
        if sim_cap < 1:
            raise ValueError(f"w_override must be >= 1 (got {w_override})")

    # Line 1: z*_0 <- FwdPass(s0); B0 <- 0.1*I; init T; MAC <- 0
    with torch.no_grad():
        context_0 = backbone.encode_context(prompt)
    if context_0.dim() == 1:
        context_0 = context_0.unsqueeze(0)

    # Per-episode RNG seeding so that multi-seed runs actually explore
    # distinct trees. ``z0_seed=None`` keeps the historical behaviour
    # (deterministic 2026 seed); ``selection_seed=None`` disables the
    # PUCT Gumbel-perturbation generator (back-compat).
    _z0_g = torch.Generator(device=device).manual_seed(
        int(z0_seed) if z0_seed is not None else 2026
    )
    z0_root = init_z0(K, d, device, _z0_g)
    _select_g: Optional[torch.Generator] = None
    if selection_seed is not None:
        _select_g = torch.Generator(device="cpu").manual_seed(int(selection_seed))

    tree = SearchTree()
    root_id = tree.new_node(prompt, z0_root, depth=0, parent_id=None, W=W_eff)

    mac_accumulated = 0.0
    total_iterations = 0
    sim_count = 0

    if faiss_context is not None:
        faiss_context.reset()

    import time as _time
    _start_t = _time.time()
    _deadline = (_start_t + wall_clock_budget_s) if wall_clock_budget_s and wall_clock_budget_s > 0 else None

    # Line 2: while MAC < tau do
    while mac_accumulated < tau_budget:
        if _deadline is not None and _time.time() > _deadline:
            break
        # Paper Table 15 sim-budget cap: when ``w_override`` is set, halt the
        # outer PUCT loop after exactly ``sim_cap`` iterations regardless of
        # the analytic FLOP budget. ``None`` preserves the historical
        # tau_budget / ACT halting path used by the integrated baselines.
        if sim_cap is not None and sim_count >= sim_cap:
            break
        sim_count += 1
        # Line 3: s <- PUCT(T, V_psi, nu_expl)
        leaf_id = _select_leaf(
            tree, nu_expl=1.0, variant=puct_variant,
            nu_temp=0.0, generator=_select_g,
        )
        leaf = tree.nodes[leaf_id]

        # Line 4: nu_A <- pi_phi(z*_s)
        z_star_s = leaf.z_star
        z_pooled = _pool_z_star(z_star_s)
        if z_pooled is None:
            z_pooled = torch.zeros(d, device=device, dtype=torch.float32)

        with torch.no_grad():
            nu, priors = meta_policy(z_pooled.to(device))
        if nu_config_mode is not None:
            nu = nu.apply_config(nu_config_mode)
        if nu_trace is not None:
            nu_trace.append(nu)

        # Line 5: MAC += LUT[pi_phi] + LUT[V_psi]
        meta_mac = 0.002e14
        mac_accumulated += meta_mac

        # Line 3 (refined): use nu_expl from policy for PUCT in subsequent iterations.
        # nu_temp now wires the meta-policy temperature into PUCT selection
        # via Gumbel noise (paper §3 Adaptive Control Operators); this is the
        # missing link that previously made all seeds collapse to identical
        # outputs.
        if leaf.depth > 0:
            leaf_id = _select_leaf(
                tree,
                nu_expl=nu.nu_expl,
                variant=puct_variant,
                nu_temp=float(nu.nu_temp),
                generator=_select_g,
            )
            leaf = tree.nodes[leaf_id]
            z_star_s = leaf.z_star
            z_pooled = _pool_z_star(z_star_s)
            if z_pooled is None:
                z_pooled = torch.zeros(d, device=device, dtype=torch.float32)
            with torch.no_grad():
                nu, priors = meta_policy(z_pooled.to(device))
            if nu_config_mode is not None:
                nu = nu.apply_config(nu_config_mode)
            if nu_trace is not None:
                nu_trace.append(nu)

        # Line 6: t <- depth(s); {z_tilde_w} <- z*_s + epsilon_w
        t = leaf.depth

        # Line 7-11: for w = 1,...,W in parallel (W_eff honours k_override)
        child_q_values: List[float] = []
        # Paper Remark 2: thread the parent's converged inverse Jacobian (if any)
        # into each child solve as the warm start.
        leaf_inv_jac = getattr(leaf, "inv_jacobian_state", None)
        # Paper §7.7: query the Hybrid KV manager for a per-leaf decision. The
        # decision call is recorded on the result regardless of whether the
        # KV-reuse fast path is wired into the active backbone, so reviewers
        # can audit that the §7.7 policy actually fires for shallow nodes.
        kv_use, kv_cached = (
            hybrid_transition_decision(leaf.depth, leaf_id, hybrid_kv_manager, backbone, leaf.text_state)
            if hybrid_kv_manager is not None
            else (False, None)
        )
        for w in range(W_eff):
            if _deadline is not None and _time.time() > _deadline:
                break
            budget_w = RuntimeBudgetState(mac_accumulated=mac_accumulated)
            r = transition(
                leaf.text_state,
                w,
                nu,
                budget_w,
                backbone,
                K=K,
                d=d,
                top_k=top_k,
                broyden_max_iter=broyden_max_iter,
                broyden_tol_min=broyden_tol_min,
                broyden_tol_max=broyden_tol_max,
                tau_flops_budget=tau_budget,
                routing_mode=routing_mode,
                max_decode_tokens=1,
                faiss_context=faiss_context if t >= 10 else None,
                parent_z_star=z_star_s,
                noise_sigma=noise_sigma,
                parent_inv_jacobian=leaf_inv_jac,
            )

            iters = r.solver_stats.get("iterations", 0)
            total_iterations += iters
            step_mac = r.solver_stats.get("flops_broyden_estimate", r.solver_stats.get("flops_used", 0.0))
            mac_accumulated += step_mac

            # Line 12: Q_w <- V_psi(z*_w); F.add(z*_w); AddChild(T, z*_w, B_w)
            z_child = r.z_star_child
            z_child_pooled = _pool_z_star(z_child)
            if z_child_pooled is None:
                z_child_pooled = torch.zeros(d, device=device)

            with torch.no_grad():
                q_w = float(critic(z_child_pooled.unsqueeze(0).to(device)).item())

            if r.prune:
                q_w = 0.0

            child_q_values.append(q_w)

            child_text = r.child_text or f"<d={t+1} w={w}>"
            child_id = tree.new_node(
                child_text, z_child, depth=t + 1, parent_id=leaf_id, W=W_eff,
            )
            # Paper Remark 2: store the converged inverse Jacobian on the child
            # so this child's own children can warm-start from it. Only the dense
            # Broyden path populates this; on the Anderson path it stays None.
            inv_jac_child = r.solver_stats.get("inv_jacobian")
            if inv_jac_child is not None:
                tree.nodes[child_id].inv_jacobian_state = inv_jac_child.detach()

        # Update priors on the expanded node (W_eff honours k_override)
        tree.nodes[leaf_id].mcts_prior = list(priors) if len(priors) == W_eff else [1.0 / W_eff] * W_eff
        tree.nodes[leaf_id].mcts_Q = child_q_values[:W_eff]

        # Line 14: BackProp(T, {Q_w})
        _backpropagate(tree, leaf_id, child_q_values)

        # Line 15-16: if MAC >= tau * nu_act then break
        if mac_accumulated >= tau_budget * nu.nu_act:
            break

    # Line 18: y_hat <- Decode(W_proj @ z*_best)
    best_id = 0
    best_q = float("-inf")
    for node in tree.nodes:
        if node.z_star is not None and node.depth > 0:
            z_p = _pool_z_star(node.z_star)
            if z_p is not None:
                with torch.no_grad():
                    v = float(critic(z_p.unsqueeze(0).to(device)).item())
                if v > best_q:
                    best_q = v
                    best_id = node.node_id

    best_z = tree.nodes[best_id].z_star

    answer = ""
    if best_z is not None and hasattr(backbone, "decode_from_z_star"):
        try:
            # Pass the original ``prompt`` so the soft-prompt prefix is
            # composed with the actual problem context. This avoids the
            # failure mode where a compute-limited W_proj produces random
            # non-sequitur tokens (e.g. 'Cultura', 'LinearLayout') on
            # math/AIME prompts because the soft-prompt alone lacks the
            # textual grounding the model needs to emit a numeric answer.
            # ``decode_from_z_star`` is backwards-compatible: backbones
            # that don't accept ``problem_text`` fall through the except
            # branch and the previous soft-prompt-only path is used.
            try:
                answer = backbone.decode_from_z_star(
                    best_z,
                    max_new_tokens=max_decode_tokens,
                    problem_text=prompt,
                )
            except TypeError:
                answer = backbone.decode_from_z_star(
                    best_z, max_new_tokens=max_decode_tokens
                )
        except Exception:
            answer = tree.nodes[best_id].text_state

    stats: Dict[str, Any] = {
        "tree_size": len(tree.nodes),
        "max_depth": max(n.depth for n in tree.nodes),
        "best_node_id": best_id,
        "best_q": best_q,
        # Sweep-driver provenance (paper Tables 13 / 15). When neither
        # override is set, ``k_override_used`` falls back to ``W`` so
        # downstream aggregation always sees the *effective* branching
        # factor; ``w_override_used`` stays ``None`` to disambiguate the
        # uncapped (FLOP-budget halting) path.
        "k_override_used": W_eff,
        "w_override_used": sim_count if sim_cap is not None else None,
        "sim_count": sim_count,
    }
    if hybrid_kv_manager is not None:
        stats["hybrid_kv"] = hybrid_kv_manager.report()
    return CtsEpisodeResult(
        answer=answer,
        best_z_star=best_z,
        tree=tree,
        total_mac=mac_accumulated,
        total_iterations=total_iterations,
        stats=stats,
    )
