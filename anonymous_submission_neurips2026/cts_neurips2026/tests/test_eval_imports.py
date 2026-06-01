"""Lightweight import checks for eval helpers (no Gemma weights)."""


def test_gemma_predict_module_imports():
    from cts.eval import gemma_predict  # noqa: F401

    assert hasattr(gemma_predict, "build_gemma_predictor")
