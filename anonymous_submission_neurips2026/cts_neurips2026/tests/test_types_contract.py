from cts.types import NuVector, RuntimeBudgetState


def test_budget_clone_independent():
    b = RuntimeBudgetState(mac_accumulated=1.0)
    c = b.clone()
    c.mac_accumulated = 2.0
    assert b.mac_accumulated == 1.0
