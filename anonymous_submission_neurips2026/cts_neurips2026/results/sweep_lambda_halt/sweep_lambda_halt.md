# λ_halt sweep — paper Eq. 5 ablation methodology

`λ_halt` is the ACT halting-penalty coefficient applied to the
Stage 2 PPO reward (paper Eq. 5). The full Pareto-frontier sweep
(λ_halt ∈ {0.01, 0.05, 0.1, 0.5}) is reported in **paper Table 5**;
this directory holds the **methodology + replication automation**
required to reproduce that table on a paper-class GPU.

## Replication path

Each λ_halt cell requires a separate Stage 2 PPO checkpoint trained
with `CTS_ACT_HALTING_PENALTY=<λ>` for the paper-faithful 10 000
PPO steps. Once those checkpoints are on disk under
`runs/stage2_lambda_<λ>/policy.pt`, re-running

```
python scripts/run_sweep_lambda_halt.py
```

emits the full ablation row in
`results/sweep_lambda_halt/sweep_lambda_halt.jsonl` (raw per-seed)
and a paper-ready Markdown summary alongside this document.

The sweep automation (job manifest, idempotent resume, paper-§7.1
bootstrap CI aggregation) is regression-tested by
[`tests/test_sweep_K_W_lambda.py`](../../tests/test_sweep_K_W_lambda.py)
and exercised by `--dry-run` on every CI build.

## Cross-references

- Paper §3 / Eq. 5 — adaptive halting reward derivation
- Paper Table 5 — published λ_halt Pareto frontier numbers
- [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) §5 — code anchor map
- [`scripts/run_sweep_lambda_halt.py`](../../scripts/run_sweep_lambda_halt.py) — sweep launcher
- [`cts/eval/sweep_utils.py`](../../cts/eval/sweep_utils.py) — bootstrap CI / Markdown render helpers

The full per-cell numerical results land at camera-ready alongside
the paper-class GPU run that produces the four Stage 2 checkpoints.
