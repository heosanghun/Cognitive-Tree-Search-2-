from cts.eval.isoflop_matcher import format_isoflop_report


def test_format_isoflop_report_includes_broyden():
    stats = {
        "iterations": 5,
        "flops_inner_once": 100.0,
        "flops_broyden_estimate": 1000.0,
        "flops_used": 100.0,
        "phi_evals_per_broyden_iter": 2,
        "converged": True,
        "residual_norm": 0.01,
    }
    r = format_isoflop_report(stats)
    assert r["flops_broyden_estimate"] == 1000.0
    assert r["broyden_iterations"] == 5


def test_format_fallback_when_broyden_missing():
    stats = {"iterations": 3, "flops_used": 10.0, "converged": True}
    r = format_isoflop_report(stats)
    assert r["flops_broyden_estimate"] == 10.0 * 3 * 2
