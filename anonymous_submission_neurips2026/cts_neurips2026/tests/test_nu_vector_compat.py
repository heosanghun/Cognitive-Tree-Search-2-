"""Tests for NuVector backward compatibility + paper naming (§2.3)."""

from cts.types import NuVector, RuntimeBudgetState


def test_nu_vector_paper_names():
    nu = NuVector(nu_val=1.5, nu_expl=2.0, nu_tol=0.3, nu_temp=0.8, nu_act=1.2)
    assert nu.nu_val == 1.5
    assert nu.nu_expl == 2.0
    assert nu.nu_tol == 0.3
    assert nu.nu_temp == 0.8
    assert nu.nu_act == 1.2


def test_nu_vector_legacy_aliases():
    nu = NuVector(nu_val=1.5, nu_expl=2.0, nu_tol=0.3, nu_temp=0.8, nu_act=1.2)
    assert nu.nu_da == 1.5
    assert nu.nu_5ht == 2.0
    assert nu.nu_ne == 0.3
    assert nu.nu_ach == 0.8
    assert nu.nu_ado_scale == 1.2


def test_budget_paper_names():
    b = RuntimeBudgetState(mac_accumulated=100.0, terminal_depth=5)
    assert b.mac_accumulated == 100.0
    assert b.terminal_depth == 5
    assert b.ado_accumulated == 100.0  # backward compat


def test_budget_clone_preserves_terminal_depth():
    b = RuntimeBudgetState(mac_accumulated=50.0, terminal_depth=3)
    c = b.clone()
    assert c.terminal_depth == 3
    c.terminal_depth = 10
    assert b.terminal_depth == 3
