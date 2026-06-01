# Config ↔ paper mapping

| Profile | Use |
|---------|-----|
| `paper_parity.yaml` | Appendix-style training lengths (10k/10k), `parallel` DEQ, `eval_deterministic`; `--tier full --config paper_parity` |

| YAML key | Paper reference |
|----------|-----------------|
| `broyden_max_iter`, `broyden_tol_*`, FP32 internal | Appendix A.2 |
| `mcts_branching_W`, `mcts_simulations_per_step` | Sec 7, Appendix A.2 |
| `lr`, `batch_size`, `gamma`, `lora_rank`, PPO steps | Appendix A.2 |
| `ppo_*`, `gae_lambda`, `entropy_coef` | Improved plan (reproducibility) |
| `stage1_openmath_n` | Sec 6.1 |
| `tau_flops_budget` | Iso-FLOP ~1e14 per query (Sec 7.3) |

**Iso-FLOP:** Canonical field names and public JSON shape: **`cts/eval/flops_contract.py`** (re-exports `format_isoflop_report` as `public_isoflop_report`). Implementations live in `cts/eval/isoflop_matcher.py`; raw stats come from `transition()`.

- **DEQ / Broyden:** `transition()` reports `flops_inner_once` and `flops_broyden_estimate` (φ-eval × iterations × 2).
- **LM decode (bench):** `cts/eval/gemma_predict.py` uses `generate()` for MATH/ARC scripts — **not** included in DEQ Iso-FLOP unless you add a separate decode budget term.
