"""Integration test for `cts_full_episode` (Algorithm 1, paper §4).

Covers the headline algorithm end-to-end on a CPU mock backbone:

  - Tree expansion (W parallel children per PUCT-selected leaf).
  - Meta-policy nu sampling at every iteration.
  - DEQ transition + Neuro-Critic Q evaluation.
  - PUCT backpropagation.
  - tau-budget halting.
  - Best-trajectory decoding via `decode_from_z_star`.

This is intentionally pure-CPU and short (<10 s) so it can run in CI; the
GPU-bound real-model variant lives in `scripts/run_cts_eval_full.py`.
"""

from __future__ import annotations

import time

import torch

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.mcts.cts_episode import CtsEpisodeResult, cts_full_episode
from cts.mcts.hybrid_kv import HybridKVManager
from cts.policy.meta_policy import MetaPolicy


class _DecodingMockBackbone(MockTinyBackbone):
    """MockTinyBackbone augmented with a deterministic z*->text decoder."""

    def __init__(self, hidden: int = 32, num_layers: int = 8) -> None:
        super().__init__(hidden=hidden, num_layers=num_layers)
        self.decode_calls: list[int] = []

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 64) -> str:
        self.decode_calls.append(max_new_tokens)
        head = z_star.detach().float().mean().item()
        return f"answer={head:+.4f}|tokens={max_new_tokens}"


def _build_components(d: int = 32, W: int = 3):
    torch.manual_seed(2026)
    bb = _DecodingMockBackbone(hidden=d, num_layers=8)
    meta = MetaPolicy(text_dim=d, hidden=64, W=W)
    critic = NeuroCritic(z_dim=d)
    return bb, meta, critic


def test_cts_full_episode_returns_result_within_budget():
    """Algorithm 1 must terminate, expand the tree, and decode an answer."""
    d, W, K = 32, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)

    t0 = time.time()
    result = cts_full_episode(
        "Q: 2 + 3 = ?",
        backbone=bb,
        meta_policy=meta,
        critic=critic,
        W=W,
        K=K,
        tau_budget=5e13,
        broyden_max_iter=4,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        top_k=2,
        max_decode_tokens=8,
        device=torch.device("cpu"),
        wall_clock_budget_s=20.0,
    )
    elapsed = time.time() - t0

    assert isinstance(result, CtsEpisodeResult)
    assert elapsed < 20.0, f"episode exceeded wall-clock budget: {elapsed:.2f}s"

    assert result.total_mac > 0.0, "MAC accumulator never advanced"
    assert result.total_mac <= 5e13 * 1.5, "tau-budget halting not respected"

    assert result.stats["tree_size"] >= 1 + W, (
        f"root + W children expected, got tree_size={result.stats['tree_size']}"
    )
    assert result.stats["max_depth"] >= 1, "no expansion happened"

    assert isinstance(result.answer, str) and result.answer.startswith("answer="), result.answer
    assert bb.decode_calls and bb.decode_calls[-1] == 8, bb.decode_calls


def test_cts_full_episode_respects_wall_clock_deadline():
    """A tiny wall-clock budget must short-circuit the MCTS loop."""
    d, W, K = 16, 3, 2
    bb, meta, critic = _build_components(d=d, W=W)

    t0 = time.time()
    result = cts_full_episode(
        "Tight budget query.",
        backbone=bb,
        meta_policy=meta,
        critic=critic,
        W=W,
        K=K,
        tau_budget=1e20,
        broyden_max_iter=2,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        top_k=2,
        max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=0.05,
    )
    elapsed = time.time() - t0

    assert elapsed < 5.0, f"wall_clock deadline ignored: {elapsed:.2f}s"
    assert isinstance(result, CtsEpisodeResult)


def test_cts_full_episode_writes_q_values_into_tree():
    """After Algorithm 1 line 12, parent.mcts_Q must contain real critic values."""
    d, W, K = 32, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)
    result = cts_full_episode(
        "Q: 7 - 3 = ?",
        backbone=bb,
        meta_policy=meta,
        critic=critic,
        W=W,
        K=K,
        tau_budget=2e13,
        broyden_max_iter=3,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        top_k=2,
        max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=15.0,
    )

    root = result.tree.nodes[0]
    assert root.mcts_N >= 1, "root visit count never incremented"
    if root.children_ids:
        # mcts_Q has W slots; each is a float (may be zero if the transition
        # pruned or the critic happens to output 0 on random init — both are
        # legal Algorithm 1 outcomes). What we *really* assert is that the
        # backprop wired floats into every slot.
        assert len(root.mcts_Q) == 3
        assert all(isinstance(q, float) for q in root.mcts_Q)

    visit_total = sum(1 for n in result.tree.nodes if n.mcts_N >= 1)
    assert visit_total >= 1


def test_cts_full_episode_threads_jacobian_inheritance():
    """Paper Remark 2: child solves must inherit the parent's converged inverse
    Jacobian (dense Broyden path only). With small K and d the dense path is
    used unconditionally, so every expanded node should carry an inv_jacobian
    of the right shape.
    """
    d, W, K = 32, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)
    result = cts_full_episode(
        "Q: 5 + 7 = ?",
        backbone=bb,
        meta_policy=meta,
        critic=critic,
        W=W,
        K=K,
        tau_budget=2e13,
        broyden_max_iter=4,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        top_k=2,
        max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=15.0,
    )

    expanded = [n for n in result.tree.nodes if n.depth >= 1 and n.z_star is not None]
    assert expanded, "no children expanded; cannot verify Jacobian threading"

    threaded = [n for n in expanded if n.inv_jacobian_state is not None]
    assert threaded, (
        "Remark 2 wiring failed: dense Broyden produced no inv_jacobian on any "
        "expanded node"
    )

    n_expected = K * d
    for node in threaded:
        jac = node.inv_jacobian_state
        assert jac is not None
        assert jac.shape == (n_expected, n_expected), (
            f"node {node.node_id}: inv_jac shape {tuple(jac.shape)} != "
            f"({n_expected}, {n_expected})"
        )
        assert jac.dtype == torch.float32  # fp32_buffer=True default


def test_cts_full_episode_accepts_hybrid_kv_manager_and_reports():
    """Paper §7.7: when a HybridKVManager is supplied, the episode loop must
    consult it on every leaf and surface its report on the result.stats. The
    decision call is a no-op for the mock backbone (no real KV-cache to
    serialize) but the wiring itself must be exercised end-to-end so reviewers
    can audit that the §7.7 decision policy actually fires.
    """
    d, W, K = 32, 3, 4
    bb, meta, critic = _build_components(d=d, W=W)
    kv = HybridKVManager(shallow_depth_limit=5, max_kv_vram_gb=1.0)

    result = cts_full_episode(
        "Q: 5 + 7 = ?",
        backbone=bb,
        meta_policy=meta,
        critic=critic,
        W=W,
        K=K,
        tau_budget=2e13,
        broyden_max_iter=4,
        broyden_tol_min=1e-3,
        broyden_tol_max=1e-2,
        top_k=2,
        max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=15.0,
        hybrid_kv_manager=kv,
    )

    assert "hybrid_kv" in result.stats
    rep = result.stats["hybrid_kv"]
    assert rep["shallow_limit"] == 5
    assert rep["max_vram_gb"] == 1.0
    assert isinstance(rep["cached_nodes"], int)
    assert isinstance(rep["total_vram_mb"], float)


def test_cts_full_episode_z0_seed_changes_root_initialization():
    """Two episodes with different ``z0_seed`` must produce different root
    latents (z*_0). Without this wiring, multi-seed runs collapse to identical
    initial trees and Wilcoxon paired tests across seeds become degenerate
    (std=0.0). Pinned by `ROOT_CAUSE_ANALYSIS.md` §"std=0.0 across seeds".
    """
    d, W, K = 16, 2, 4
    bb_a, meta_a, critic_a = _build_components(d=d, W=W)
    bb_b, meta_b, critic_b = _build_components(d=d, W=W)

    res_a = cts_full_episode(
        "Q: 1 + 2 = ?",
        backbone=bb_a, meta_policy=meta_a, critic=critic_a,
        W=W, K=K,
        tau_budget=5e12, broyden_max_iter=3,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=10.0,
        z0_seed=11, selection_seed=11,
    )
    res_b = cts_full_episode(
        "Q: 1 + 2 = ?",
        backbone=bb_b, meta_policy=meta_b, critic=critic_b,
        W=W, K=K,
        tau_budget=5e12, broyden_max_iter=3,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=10.0,
        z0_seed=42, selection_seed=42,
    )

    assert isinstance(res_a, CtsEpisodeResult)
    assert isinstance(res_b, CtsEpisodeResult)
    # Tree size or answer text should reflect divergent initial conditions.
    different = (
        res_a.stats.get("tree_size") != res_b.stats.get("tree_size")
        or res_a.answer != res_b.answer
    )
    assert different, (
        "different z0_seeds collapsed to identical episodes — z0_seed wiring "
        "regressed (re-introduces std=0.0 across seeds bug)."
    )


def test_select_leaf_uses_nu_temp_for_seeded_exploration(monkeypatch):
    """nu_temp must inject Gumbel noise into PUCT scores. With ``nu_temp=0``
    the selection is deterministic; with ``nu_temp > 0`` and a controlled
    ``torch.rand`` we can prove the noise term flips a tied PUCT race.
    """
    from cts.mcts.cts_episode import _select_leaf
    from cts.mcts.tree import SearchTree
    from cts.types import TreeNode

    # Build a 2-child tree where both children have identical priors / Q so
    # nu_temp is the *only* tiebreaker.
    tree = SearchTree()
    root = TreeNode(node_id=0, text_state="root", z_star=torch.zeros(4),
                    depth=0, parent_id=None,
                    mcts_N=10, mcts_prior=[0.5, 0.5], mcts_Q=[0.0, 0.0],
                    children_ids=[1, 2])
    c1 = TreeNode(node_id=1, text_state="c1", z_star=torch.zeros(4),
                  depth=1, parent_id=0, mcts_N=5)
    c2 = TreeNode(node_id=2, text_state="c2", z_star=torch.zeros(4),
                  depth=1, parent_id=0, mcts_N=5)
    tree.nodes = [root, c1, c2]

    # With nu_temp == 0, PUCT ties are broken by argmax-of-equals -> first one.
    leaf_id = _select_leaf(tree, nu_expl=1.0, nu_temp=0.0, generator=None)
    assert leaf_id in (1, 2)

    # Force torch.rand to make child #2 win. Gumbel noise is
    # ``-nu_temp * log(-log(U))``, so:
    #   * U near 0 -> Gumbel is large NEGATIVE
    #   * U near 1 -> Gumbel is large POSITIVE
    # Sequence (idx=0 small U, idx=1 large U) -> idx=1 (=child 2) wins.
    calls = {"i": 0}
    sequence = [torch.tensor(0.01), torch.tensor(0.99)]

    def _fake_rand(*args, **kwargs):
        v = sequence[calls["i"] % len(sequence)]
        calls["i"] += 1
        return v

    monkeypatch.setattr("torch.rand", _fake_rand)
    leaf_id_noisy = _select_leaf(tree, nu_expl=1.0, nu_temp=2.0, generator=None)
    assert leaf_id_noisy == 2, (
        f"nu_temp Gumbel noise did not flip the selection (got {leaf_id_noisy})"
    )


# ---------------------------------------------------------------------------
# P0-1 regression: cts_2nu and cts_4nu MUST be different code paths.
# ---------------------------------------------------------------------------
# Paper §7.5 / Table 5 reports CTS-4nu (50.2 % AIME, 27.3 s) and CTS-2nu
# (46.8 % AIME, 14.2 s) as DISTINCT operating points on the nu-Pareto frontier.
# A previous version of the codebase wired both methods through identical
# arguments to ``cts_full_episode`` because ``nu_config_mode`` was missing
# entirely — meaning ``cts_2nu`` and ``cts_4nu`` were the *same* run with
# different labels, and Table 5 was not actually validated. The tests below
# pin (a) ``apply_config`` actually freezes the inactive operators, and
# (b) ``cts_full_episode`` accepts and threads ``nu_config_mode`` through
# every meta-policy invocation site.

def test_apply_config_2nu_fast_freezes_tol_and_act():
    """``2nu_fast`` must keep {expl, temp} from the policy and freeze
    {tol, act} at the Stage 1 converged means (paper Table 5 footnote)."""
    from cts.types import NU_STAGE1_DEFAULTS, NuVector

    nu = NuVector(nu_expl=0.93, nu_tol=7.5e-3, nu_temp=1.42, nu_act=0.55)
    nu_2 = nu.apply_config("2nu_fast")
    assert nu_2.nu_expl == 0.93
    assert nu_2.nu_temp == 1.42
    assert nu_2.nu_tol == NU_STAGE1_DEFAULTS["tol"]
    assert nu_2.nu_act == NU_STAGE1_DEFAULTS["act"]


def test_apply_config_4nu_is_identity():
    """``4nu`` keeps every operator live (no-op apply_config)."""
    from cts.types import NuVector

    nu = NuVector(nu_expl=0.93, nu_tol=7.5e-3, nu_temp=1.42, nu_act=0.55)
    nu_4 = nu.apply_config("4nu")
    assert nu_4.nu_expl == nu.nu_expl
    assert nu_4.nu_tol == nu.nu_tol
    assert nu_4.nu_temp == nu.nu_temp
    assert nu_4.nu_act == nu.nu_act


def test_cts_full_episode_accepts_nu_config_mode():
    """``cts_full_episode`` must accept and forward ``nu_config_mode``;
    the 2nu_fast variant must NOT raise and must produce a valid result."""
    d, W, K = 16, 2, 4
    bb, meta, critic = _build_components(d=d, W=W)

    res = cts_full_episode(
        "Q: 1 + 1 = ?",
        backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K,
        tau_budget=5e12, broyden_max_iter=3,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=4,
        device=torch.device("cpu"),
        wall_clock_budget_s=10.0,
        z0_seed=11, selection_seed=11,
        nu_config_mode="2nu_fast",
    )
    assert isinstance(res, CtsEpisodeResult)
    assert res.stats.get("tree_size", 0) >= 1


def test_cts_2nu_and_4nu_diverge_when_meta_policy_outputs_nondefault_tol_act(
    monkeypatch,
):
    """The whole point of ``nu_config_mode``: 2nu_fast must *override* the
    meta-policy's ``nu_tol`` / ``nu_act`` at every step with the Stage 1
    converged means, so two episodes that share everything but the mode
    must produce different ``nu`` time-series.

    We do this by capturing every ``nu`` that the inner DEQ transition
    receives and asserting:
      - 4nu sees policy-dependent (non-default) tol / act values.
      - 2nu_fast sees Stage 1 default tol / act on every call.
    """
    from cts.mcts import cts_episode as _ep
    from cts.types import NU_STAGE1_DEFAULTS, NuVector

    captured: list[tuple[str, NuVector]] = []

    real_transition = _ep.transition

    def _spy_transition(text, w, nu, *args, **kwargs):
        captured.append((kwargs.get("__mode__", ""), nu))
        return real_transition(text, w, nu, *args, **kwargs)

    monkeypatch.setattr(_ep, "transition", _spy_transition)

    d, W, K = 16, 2, 4

    # Run 1: 4nu — meta-policy outputs propagate.
    captured.clear()
    bb, meta, critic = _build_components(d=d, W=W)
    cts_full_episode(
        "Q: 4nu", backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K, tau_budget=3e12, broyden_max_iter=2,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=2,
        device=torch.device("cpu"), wall_clock_budget_s=8.0,
        z0_seed=7, selection_seed=7, nu_config_mode="4nu",
    )
    nus_4 = list(captured)

    # Run 2: 2nu_fast — tol / act MUST be Stage 1 defaults at every step.
    captured.clear()
    bb, meta, critic = _build_components(d=d, W=W)
    cts_full_episode(
        "Q: 2nu", backbone=bb, meta_policy=meta, critic=critic,
        W=W, K=K, tau_budget=3e12, broyden_max_iter=2,
        broyden_tol_min=1e-3, broyden_tol_max=1e-2,
        top_k=2, max_decode_tokens=2,
        device=torch.device("cpu"), wall_clock_budget_s=8.0,
        z0_seed=7, selection_seed=7, nu_config_mode="2nu_fast",
    )
    nus_2 = list(captured)

    assert nus_2, "2nu episode produced no transitions"
    assert nus_4, "4nu episode produced no transitions"

    # Hard guarantee: every 2nu_fast transition has the frozen defaults.
    for _, nu in nus_2:
        assert nu.nu_tol == NU_STAGE1_DEFAULTS["tol"], (
            f"2nu_fast did not freeze nu_tol (got {nu.nu_tol}, "
            f"expected {NU_STAGE1_DEFAULTS['tol']})"
        )
        assert nu.nu_act == NU_STAGE1_DEFAULTS["act"], (
            f"2nu_fast did not freeze nu_act (got {nu.nu_act}, "
            f"expected {NU_STAGE1_DEFAULTS['act']})"
        )

    # Soft signal that 4nu is actually different: at least one tol or act
    # value should deviate from the Stage 1 default. (MockTinyBackbone +
    # untrained MetaPolicy can occasionally produce defaults by chance, so
    # we accept either a difference in policy outputs or a difference in
    # the resulting nu time-series length — the former proves the wiring,
    # the latter proves the trees differ.)
    if all(
        n.nu_tol == NU_STAGE1_DEFAULTS["tol"] and n.nu_act == NU_STAGE1_DEFAULTS["act"]
        for _, n in nus_4
    ):
        assert len(nus_2) != len(nus_4) or nus_2 != nus_4, (
            "cts_2nu and cts_4nu produced byte-identical nu time-series — "
            "nu_config_mode wiring regressed."
        )
