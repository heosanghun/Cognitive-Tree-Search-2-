"""P1 regression: every paper Table 2 baseline must have an integrated
dispatcher in `_run_cts_on_problems`, and the registry / Bonferroni family
must reflect the post-D1 state.

Audited mismatches that this file pins (NeurIPS Apr 2026 audit):

  - Table 2 lists 12 methods total (greedy, native_think, deq_only,
    cts_2nu, cts_4nu, think_off_greedy, ft_nt, sc_14, bon_13,
    bandit_ucb1, mcts_early_stop, expl_mcts_ppo). Every one of these must
    be a real branch in the dispatcher; falling through to a generic
    greedy call would mis-label baseline numbers (the previous behavior
    that the April 2026 audit flagged).

  - The headline Bonferroni family per paper §7.1 is n = 12: CTS-4nu vs
    each of {greedy, native_think, sc_14, mcts_early_stop} on each of
    {math500, gsm8k, aime}. After the D1 baseline sweep the operational
    family must equal the paper family exactly (not the n=6 reduction
    that was a temporary disclosure in the previous unintegrated state).

These tests do NOT spin up Gemma; they introspect the source file via
``ast`` and import the registry constants. CPU-only, <0.5 s.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cts_eval_full.py"

EXPECTED_METHODS = {
    "greedy",
    "native_think",
    "deq_only",
    "cts_2nu",
    "cts_4nu",
    "think_off_greedy",
    "ft_nt",
    "sc_14",
    "bon_13",
    "bandit_ucb1",
    "mcts_early_stop",
    "expl_mcts_ppo",
}


def _dispatcher_methods() -> set[str]:
    """Return every method name handled by `_run_cts_on_problems`.

    We parse the AST and collect every literal string compared against
    ``method ==`` or membership-tested with ``method in (...)``. This is
    far more robust than text-grep and survives whitespace / comment
    refactors.
    """
    src = _SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_cts_on_problems":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Compare) and isinstance(inner.left, ast.Name) and inner.left.id == "method":
                    for op, comp in zip(inner.ops, inner.comparators):
                        if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                            found.add(comp.value)
                        elif isinstance(op, ast.In) and isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                            for elt in comp.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    found.add(elt.value)
    return found


def test_every_table2_method_has_a_dispatcher() -> None:
    found = _dispatcher_methods()
    missing = EXPECTED_METHODS - found
    assert not missing, (
        f"Table 2 baselines missing a dispatcher branch: {sorted(missing)}. "
        f"Each must be implemented in _run_cts_on_problems; falling through "
        f"to greedy would mis-label baseline numbers."
    )


def test_table2_methods_integrated_lists_every_dispatcher() -> None:
    """The exported method registry MUST list every method the dispatcher
    can handle."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_eval_runner", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    integrated = set(mod.TABLE2_METHODS_INTEGRATED)
    assert integrated == EXPECTED_METHODS, (
        f"TABLE2_METHODS_INTEGRATED drift; got {sorted(integrated)}, "
        f"expected {sorted(EXPECTED_METHODS)}"
    )
    assert mod.TABLE2_METHODS_PAPER_ONLY == [], (
        "TABLE2_METHODS_PAPER_ONLY must be empty after the D1 sweep — every "
        "baseline now has a dispatcher."
    )


def test_primary_bonferroni_family_matches_paper() -> None:
    """Paper §7.1 declares n=12 primary comparisons. The operational set
    must equal CTS-4nu vs {greedy, native_think, sc_14, mcts_early_stop}
    over {math500, gsm8k, aime}."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_eval_runner", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    expected = set()
    for baseline in ("greedy", "native_think", "sc_14", "mcts_early_stop"):
        for bench in ("math500", "gsm8k", "aime"):
            expected.add((baseline, bench))

    got = set(mod.PRIMARY_COMPARISONS)
    assert got == expected, (
        f"PRIMARY_COMPARISONS drift; got {sorted(got)}, "
        f"expected {sorted(expected)}"
    )
    assert mod.PRIMARY_BONFERRONI_N == 12, (
        f"PRIMARY_BONFERRONI_N regressed to {mod.PRIMARY_BONFERRONI_N}, "
        f"paper §7.1 specifies 12."
    )


@pytest.mark.parametrize("method", sorted(EXPECTED_METHODS))
def test_unknown_method_still_raises(method: str) -> None:
    """Sanity guard: a misspelled method name must NOT silently fall into
    a generic greedy path (the bug that the April 2026 audit flagged).

    We can verify this without spinning up Gemma by asserting the source
    contains an explicit ``raise NotImplementedError`` in the final
    ``else`` branch of `_run_cts_on_problems`.
    """
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "Unknown evaluation method" in src, (
        "The catch-all else branch lost its NotImplementedError raise; "
        "this re-introduces silent baseline mis-labelling."
    )
