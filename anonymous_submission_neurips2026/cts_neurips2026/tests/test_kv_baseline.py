from cts.baselines.mcts_kv_baseline import KVRetentionConfig, estimate_mcts_kv_peak_gb


def test_kv_grows_with_depth():
    a = estimate_mcts_kv_peak_gb(1)
    b = estimate_mcts_kv_peak_gb(10)
    assert b > a


def test_kv_config_override():
    cfg = KVRetentionConfig(tokens_per_depth_step=128)
    g = estimate_mcts_kv_peak_gb(2, cfg)
    assert g > 0
