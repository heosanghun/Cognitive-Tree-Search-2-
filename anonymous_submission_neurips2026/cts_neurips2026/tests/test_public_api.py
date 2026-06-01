"""Lock-in test for ``cts.__all__`` &mdash; ensures every advertised public
symbol is importable and resolves to the right kind of object.

If a future commit deletes / renames a paper-cited public symbol, this
test fails before the README's "Paper &harr; Code Mapping" table goes stale.
"""

from __future__ import annotations

import importlib
import inspect


EXPECTED_API = {
    # core dataclasses (paper Sections 4.1, 4.3)
    "NuVector": "class",
    "NuConfigMode": "literal",
    "RuntimeBudgetState": "class",
    "TransitionResult": "class",
    "TreeNode": "class",
    # Algorithm 1 + DEQ (paper Sections 4.1-4.2)
    "cts_full_episode": "function",
    "puct_score": "function",
    "select_action": "function",
    "transition": "function",
    "transition_batch": "function",
    # policy / critic (paper Section 4.1)
    "MetaPolicy": "class",
    "NeuroCritic": "class",
    # latent context (paper Section 4.4)
    "LatentContextWindow": "class",
    # hybrid kv (paper Section 7.7)
    "HybridKVManager": "class",
    "hybrid_transition_decision": "function",
    # reward (paper Eq. 5)
    "paper_reward": "function",
    # statistics (paper Section 7.1)
    "bootstrap_ci": "function",
    "wilcoxon_signed_rank": "function",
    "bonferroni_correct": "function",
}


def test_import_does_not_load_torch_compile_or_gemma():
    """Importing the package must not eagerly load the Gemma backbone."""
    import sys
    sys.modules.pop("cts", None)
    cts = importlib.import_module("cts")
    assert hasattr(cts, "__version__")
    # The Gemma adapter is lazy: it's not pulled in by the top-level import.
    assert "cts.backbone.gemma_adapter" not in sys.modules


def test_all_advertised_symbols_resolve():
    cts = importlib.import_module("cts")
    for name in EXPECTED_API:
        assert hasattr(cts, name), f"public API symbol missing: {name}"


def test_all_list_matches_expected_set_exactly():
    cts = importlib.import_module("cts")
    declared = set(cts.__all__) - {"__version__"}
    expected = set(EXPECTED_API)
    missing = expected - declared
    extra = declared - expected
    assert not missing, f"public API regressed; missing: {sorted(missing)}"
    assert not extra, f"public API has undocumented symbols: {sorted(extra)}"


def test_classes_resolve_to_classes():
    cts = importlib.import_module("cts")
    for name, kind in EXPECTED_API.items():
        if kind != "class":
            continue
        obj = getattr(cts, name)
        assert inspect.isclass(obj), f"{name} should be a class, got {type(obj)}"


def test_functions_resolve_to_callables():
    cts = importlib.import_module("cts")
    for name, kind in EXPECTED_API.items():
        if kind != "function":
            continue
        obj = getattr(cts, name)
        assert callable(obj), f"{name} should be callable"


def test_paper_reward_eq5_signature_unchanged():
    """Eq. 5 signature stability: any change to the call pattern would
    invalidate the reproducibility instructions in REVIEWER_FAQ + README."""
    from cts import paper_reward
    sig = inspect.signature(paper_reward)
    assert list(sig.parameters) == ["correct", "terminal_depth", "lambda_halt"]
    assert sig.parameters["lambda_halt"].default == 0.05  # paper Table 4
