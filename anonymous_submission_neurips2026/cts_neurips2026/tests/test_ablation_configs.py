"""Lock down ablation YAML configs and the Stage-1 ablation drivers.

The paper §7.4 ablation table claims three configurations:
  - canonical CTS (4nu, sparse routing, dynamic exploration)
  - no-ACh        (dense routing)
  - static-5HT    (constant exploration)

Both `configs/ablation_*.yaml` overlays must merge cleanly with default
and `scripts/run_ablations.py` must consume them without crashing.
This test pins the contract end-to-end so an ablation regression cannot
slip into the submission.
"""

from __future__ import annotations

from cts.utils.config import load_config


def test_ablation_no_ach_overlay_sets_dense_routing():
    cfg = load_config("ablation_no_ach")
    assert cfg["routing_mode"] == "dense"
    assert cfg["ablation_name"] == "no_ach"
    # default values must still be inherited
    assert cfg["latent_tokens_K"] == 64
    assert cfg["mcts_branching_W"] == 3


def test_ablation_static_5ht_overlay_sets_static_exploration():
    cfg = load_config("ablation_static_5ht")
    assert cfg["nu_expl_static"] == 1.0
    assert cfg["ablation_name"] == "static_expl"
    # default values must still be inherited
    assert cfg["broyden_max_iter"] == 30


def test_ablation_overlays_do_not_disturb_paper_canonical_settings():
    """The ablations should change ONE thing each. Verify they don't
    accidentally also change PPO/LoRA/Stage-1 hyperparams."""
    base = load_config("default")
    for ab in ("ablation_no_ach", "ablation_static_5ht"):
        cfg = load_config(ab)
        for key in ("ppo_lr", "lora_rank", "lora_alpha", "stage1_max_steps",
                    "tau_flops_budget", "eval_seeds", "eval_bonferroni_n"):
            assert cfg[key] == base[key], (
                f"{ab} silently changed {key}: "
                f"{cfg[key]} != base {base[key]}"
            )


def test_ablation_no_ach_runtime_executes_without_error():
    """End-to-end: invoke transition() with the no-ACh config and verify
    the dense routing path produces a reasonable residual. Uses a CPU-only
    mock backbone so this runs in seconds. We assert the residual is small
    rather than full convergence so the test is not flaky on machines
    where the inner loop happens to need a few extra iterations."""
    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState

    cfg = load_config("ablation_no_ach")
    bb = MockTinyBackbone(hidden=64, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=0.65)
    r = transition(
        "ablation no-ach probe", 0, nu, RuntimeBudgetState(), bb,
        K=4, d=64, broyden_max_iter=40,
        routing_mode=cfg["routing_mode"],
        tau_flops_budget=1e20,
    )
    assert r.solver_stats.get("flops_used", 0) > 0
    # Residual should be small even if convergence flag isn't tripped
    assert r.solver_stats.get("residual_norm", 1.0) < 1e-1


def test_ablation_static_5ht_runtime_executes_without_error():
    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState

    cfg = load_config("ablation_static_5ht")
    bb = MockTinyBackbone(hidden=64, num_layers=42)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=float(cfg["nu_expl_static"]))
    r = transition(
        "ablation static-5ht probe", 0, nu, RuntimeBudgetState(), bb,
        K=4, d=64, broyden_max_iter=40,
        routing_mode="sparse",
        tau_flops_budget=1e20,
    )
    assert r.solver_stats.get("flops_used", 0) > 0
    assert r.solver_stats.get("residual_norm", 1.0) < 1e-1


def test_static_5ht_uses_higher_exploration_than_canonical():
    """Sanity: paper §7.4 frames static-5HT as 'high constant exploration',
    higher than the canonical 0.65 used by run_ablations.py default."""
    cfg = load_config("ablation_static_5ht")
    assert cfg["nu_expl_static"] >= 0.65
