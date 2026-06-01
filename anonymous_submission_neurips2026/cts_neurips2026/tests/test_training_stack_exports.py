"""Smoke: training stack entrypoints import (no Gemma load)."""


def test_stage1_stage2_training_callables_exist():
    from cts.train.stage1_openmath_train import run_stage1_openmath_training
    from cts.train.stage2_ppo_train import run_stage2_math_ppo

    assert callable(run_stage1_openmath_training)
    assert callable(run_stage2_math_ppo)
