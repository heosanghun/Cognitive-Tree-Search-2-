"""Pin paper-claimed hyperparameters in configs/{default,paper_parity}.yaml.

If a future commit silently changes any of these, the test fails and the
NeurIPS reviewer can immediately see the divergence between repo-as-shipped
and the paper. This is far more reliable than human comment-checking.

Each assertion is annotated with the exact paper section the value comes
from, so reviewers can audit the table by reading the test file alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    path = REPO_ROOT / "configs" / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_default_yaml_dt_is_bfloat16():
    """Paper §6: bf16 mixed-precision training."""
    assert _load("default")["dtype_weights"] == "bfloat16"


def test_default_yaml_broyden_settings():
    """Paper §4.2 / §5.2 / Table 1: rank-16 L-Broyden, 30 max iter."""
    cfg = _load("default")
    assert cfg["broyden_max_iter"] == 30
    assert cfg["broyden_memory_limit"] == 16
    assert cfg["broyden_fp32_buffer"] is True


def test_default_yaml_latent_K_64():
    """Paper §4.2: K=64 latent tokens."""
    cfg = _load("default")
    assert cfg["latent_tokens_K"] == 64
    assert cfg["soft_thought_K"] == 64  # legacy alias must agree


def test_default_yaml_mcts_branching_W_3():
    """Paper §4.1: W=3 MCTS branching factor."""
    assert _load("default")["mcts_branching_W"] == 3


def test_default_yaml_ppo_table4_hyperparams():
    """Paper Table 4 PPO hyperparameters."""
    cfg = _load("default")
    assert cfg["ppo_lr"] == 3.0e-5
    assert cfg["ppo_clip_epsilon"] == 0.2
    assert cfg["gae_lambda"] == 0.95
    assert cfg["entropy_coef"] == 0.01
    assert cfg["value_loss_coef"] == 0.5
    assert cfg["max_grad_norm"] == 1.0


def test_default_yaml_reward_eq5():
    """Paper Eq. 5: lambda_halt = 0.05, gamma = 0.99."""
    cfg = _load("default")
    assert cfg["act_halting_penalty"] == 0.05
    assert cfg["discount_gamma"] == 0.99
    assert cfg["lambda_ado_penalty"] == 0.05  # legacy alias


def test_default_yaml_lora_app_I():
    """Paper App. I: LoRA rank=8, alpha=16, target=q/v/o_proj."""
    cfg = _load("default")
    assert cfg["lora_rank"] == 8
    assert cfg["lora_alpha"] == 16
    assert sorted(cfg["lora_target"]) == ["o_proj", "q_proj", "v_proj"]


def test_default_yaml_data_volumes():
    """Paper §6.1, §6.2: 10K OpenMath, 5K MATH/AIME, 800 PPO episodes."""
    cfg = _load("default")
    assert cfg["stage1_openmath_n"] == 10000
    assert cfg["stage2_math_prompts_n"] == 5000
    assert cfg["ppo_episodes"] == 800


def test_default_yaml_stage1_max_steps_5000():
    """Paper §6.1: '10,000 examples for 5,000 steps'."""
    assert _load("default")["stage1_max_steps"] == 5000


def test_default_yaml_routing_act():
    """Paper §5.2 / §4.3: top-3 modules, tau=1e14 MAC budget."""
    cfg = _load("default")
    assert cfg["top_k_modules"] == 3
    # YAML may parse 1.0e14 as either float or string depending on
    # YAML 1.1 vs 1.2 implementation; production code casts via
    # float(cfg.get("tau_flops_budget", 1e14)). Test the same way.
    assert float(cfg["tau_flops_budget"]) == pytest.approx(1.0e14)


def test_default_yaml_stat_protocol_section_7_1():
    """Paper §7.1: 5 seeds, 95% CI, Bonferroni n=12."""
    cfg = _load("default")
    assert cfg["eval_seeds"] == 5
    assert cfg["eval_ci_level"] == 0.95
    assert cfg["eval_bonferroni_n"] == 12


def test_default_yaml_stage1_lambda_lm_0_1():
    """Paper §6: lambda_LM = 0.1 (limits PPL increase to 0.4 nats)."""
    assert _load("default")["stage1_lambda_lm"] == 0.1


def test_default_yaml_faiss_section_4_4():
    """Paper §4.4: top-3 ancestral retrieval, active when t > 10."""
    cfg = _load("default")
    assert cfg["faiss_enabled"] is True
    assert cfg["faiss_retrieval_k"] == 3
    assert cfg["faiss_min_steps"] == 10


def test_default_yaml_native_think_off():
    """Paper §7.1: enable_thinking=False for the canonical evaluation."""
    assert _load("default")["enable_thinking"] is False


# ----- paper_parity.yaml ----------------------------------------------------

def test_paper_parity_overlay_does_not_overtrain_stage1():
    """paper_parity.yaml is supposed to be paper-canonical; stage1_max_steps
    must equal the paper's 5,000 steps, NOT a longer over-train value."""
    overlay = _load("paper_parity")
    assert overlay["stage1_max_steps"] == 5000, (
        "paper_parity.yaml claims to be the paper-aligned profile but "
        "stage1_max_steps != 5000 (paper §6.1)."
    )


def test_paper_parity_label_is_versioned():
    """The paper_parity overlay should declare an explicit version label
    so reviewers can match an artifact ZIP to a config snapshot."""
    overlay = _load("paper_parity")
    assert "paper_protocol_label" in overlay
    assert overlay["paper_protocol_label"].startswith("paper_parity_")


def test_paper_parity_keeps_paper_broyden_max_iter():
    """Paper App. A.2: Broyden max iter = 30."""
    assert _load("paper_parity")["broyden_max_iter"] == 30


def test_paper_parity_uses_parallel_deq_map():
    """Paper §5.2: sparse parallel modules in inner DEQ map (GPU-heavy)."""
    overlay = _load("paper_parity")
    assert overlay["cts_deq_map_mode"] == "parallel"
    assert overlay["stage2_parallel_map"] is True
