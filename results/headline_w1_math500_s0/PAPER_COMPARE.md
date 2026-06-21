# Paper vs Local Comparison &mdash; Table 2

Source: `D:/AI/cts/results/headline_w1_math500_s0/table2_results.json`

Each cell shows **local mean&pm;std (n samples)** in the first row and the
**paper headline** plus the **gap (local &minus; paper)** in subsequent rows.
All values are accuracy in percent; n is the number of (seed x problem)
samples that succeeded.

| Method | MATH-500 | GSM8K | AIME 2026 | ARC-AGI-Text | HumanEval |
|---|---|---|---|---|---|
| Greedy | 0.0&pm;0.0 (n=0) | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 |
| &nbsp;&nbsp;_gap_ | -45.2 | &mdash; | &mdash; | &mdash; | &mdash; |
| SC@14 | 0.0&pm;0.0 (n=0) | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 59.3&pm;0.7 | 84.2&pm;0.5 | 34.8&pm;0.9 | 52.4&pm;0.8 | 65.2&pm;0.6 |
| &nbsp;&nbsp;_gap_ | -59.3 | &mdash; | &mdash; | &mdash; | &mdash; |
| Native Think | 0.0&pm;0.0 (n=0) | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 57.0&pm;0.6 | 82.4&pm;0.4 | 42.5&pm;0.9 | 50.1&pm;0.7 | 63.3&pm;0.5 |
| &nbsp;&nbsp;_gap_ | -57.0 | &mdash; | &mdash; | &mdash; | &mdash; |
| MCTS (Early Stop) | 0.0&pm;0.0 (n=0) | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 56.5&pm;0.9 | 81.2&pm;0.7 | 38.4&pm;0.8 | 48.1&pm;1.0 | 62.5&pm;0.7 |
| &nbsp;&nbsp;_gap_ | -56.5 | &mdash; | &mdash; | &mdash; | &mdash; |
| CTS-4nu (Ours) | 0.0&pm;0.0 (n=0) | &mdash; | &mdash; | &mdash; | &mdash; |
| &nbsp;&nbsp;_paper_ | 64.1&pm;0.8 | 88.4&pm;0.5 | 50.2&pm;1.1 | 57.8&pm;0.9 | 69.6&pm;0.7 |
| &nbsp;&nbsp;_gap_ | -64.1 | &mdash; | &mdash; | &mdash; | &mdash; |

## Notes

- Paper headlines are taken from Table 2 of the NeurIPS 2026 submission
  (Gemma 4 E4B backbone, 5 seeds, &le; 1e14 MACs, 95% bootstrap CI).
- **Baseline coverage disclosure**: the single-GPU snapshot integrates
  only `greedy`, `native_think`, `cts_2nu`, `cts_4nu`, `deq_only`. Paper
  baselines `sc_14` and `mcts_early_stop` are rendered above as
  paper-only reference numbers; the corresponding local rows will read
  `&mdash;` because `_run_cts_on_problems` raises `NotImplementedError`
  on those names rather than silently producing greedy-equivalent
  numbers. See README.md "Implementation Status" for full disclosure.
- The operational primary Bonferroni family in this snapshot is therefore
  reduced to **n=6** (CTS-4nu vs {greedy, native_think} x
  {math500, gsm8k, aime}) rather than the paper's n=12. The paper's
  full n=12 family is reproducible only after the missing baselines are
  added (multi-GPU paper-scale run).
- Local re-runs in this repo currently use a reduced wall-clock and MAC
  budget (CTS_EVAL_TAU_CAP=1e13, CTS_EVAL_EPISODE_TIMEOUT=180s) so
  absolute accuracy is expected to be lower than the paper headline; the
  *relative ordering* across methods is the headline reproducibility
  signal.
