from cts.eval.isoflop_matcher import estimate_sparse_step_flops, load_mac_per_module


def test_sparse_flops_positive():
    macs = load_mac_per_module()
    w = [1.0 / len(macs)] * len(macs)
    f = estimate_sparse_step_flops(w, macs)
    assert f > 0
