# CTS Framework Hot-Path Optimization — 2026-07-17

Goal: bring the per-episode CTS wall-clock (previously observed at ~17–27 s
for the controlled reference workload) under **10 s** *without changing search
behaviour or answer quality*.

All numbers below were produced by `scripts/bench_episode_perf.py` on a
4-core Intel Xeon @ 2.80 GHz (CPU-only, torch 2.13.0), CPU
`MockTinyBackbone`, paper-default episode invocation mirrored from
`scripts/run_cts_eval_full.py` (W = 3, broyden_max_iter = 20, FAISS context,
per-episode seeds). Baseline = commit `4f41d3f` (pre-optimization HEAD) run
from an unmodified git worktree; Optimized = this branch. Raw per-episode
JSON is in [`raw/`](raw/).

## 1. Wall-clock results

### Controlled reference workload (K = 64, d = 64, 2 simulations, interleaved runs)

| Round | Baseline (HEAD) | Optimized | Speedup |
|:-----:|:---------------:|:---------:|:-------:|
| r1 | 25.55 s | 0.78 s | 32.8× |
| r2 | 18.93 s | 0.79 s | 24.0× |
| r3 | 17.40 s | 0.75 s | 23.2× |
| **mean** | **20.63 s** | **0.77 s** | **26.7×** |

### Full eval-protocol episode (K = 64, d = 64, τ = 10¹³, 180 s wall cap — the `CTS_EVAL_TAU_CAP=1e13` re-experiment protocol)

| | Baseline (HEAD) | Optimized |
|:--|:--|:--|
| Episode wall-clock | **does not complete** — host OOM-killed (~14 GB RSS growth in <30 s on a 16 GB box) | **9.11 s** |
| Simulations / tree size / DEQ iterations | — | 30 sims, 91 nodes, 1,137 Broyden iterations |

The τ-driven episode that previously could not even finish on a 16 GB host
now completes in **9.11 s < 10 s**. The controlled 2-simulation workload
(the ~17–27 s regime) dropped to **0.77 s**.

## 2. Behaviour-equivalence verification

Same benchmark harness, both code versions, per-(seed, episode) deterministic
RNG. Every discrete outcome of the search is identical between baseline and
optimized code:

| Config | Episodes | Answer | Tree size | Sim count | Max depth | Total DEQ iterations |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| K=8, d=32, 6 sims | 3/3 identical | identical | identical | identical | identical | 174/179/173 = identical |
| K=16, d=64, 4 sims | 3/3 identical | identical | identical | identical | identical | 147/156/150 = identical |
| K=64, d=64, 2 sims | identical | `answer=+0.015833` both | 7 both | 2 both | 2 both | 84 both |

The only non-identical artifact is the low-order float rounding of z\*
(max |Δz| ≈ 1e-7 at tol = 1e-6, verified separately against an inline
re-implementation of the old solver), which is below the solver tolerance and
does not change any PUCT selection, critic ranking, halting decision, or
decoded answer.

`pytest tests/` on the optimized tree: same pass set as the pre-optimization
baseline (envir­onment-dependent failures aside — see §4).

## 3. What was changed (and why it is behaviour-preserving)

1. **`cts/deq/broyden_forward.py` — dense L-Broyden now maintains the
   *inverse* Jacobian H = B⁻¹ via Sherman–Morrison** instead of storing B and
   calling `torch.linalg.solve` (O(n³), n = K·d = 4096 → ~6.9·10¹⁰ FLOPs *per
   iteration*) every step. In exact arithmetic the iterates are identical:
   `step = −H·F(z) == solve(B, −F(z))` and
   `H' = H − (Hy − s)(sᵀH)/(sᵀHy) == (B + (y − Bs)sᵀ/sᵀs)⁻¹`.
   The rank-1 update is applied in place (`addr_`), and the two dead
   per-iteration history lists (`update_s`/`update_y`, never read) were
   removed. `jacobian_state` now genuinely stores the inverse Jacobian —
   which is what the paper's Remark 2 ("inverse Jacobian inheritance") and
   the field names (`inv_jacobian`, `parent_inv_jacobian`) always claimed.
2. **`broyden_fixed_point` runs under `torch.no_grad()`.** No CTS code path
   differentiates through the solver loop (Stage 1 trains with the IFT
   surrogate on one explicit φ step; Stage 2 PPO detaches z\* — that is the
   point of the DEQ/IFT formulation). Previously every per-iteration
   n×n tensor was retained by autograd through the episode tree's `z_star`
   references (~67 MB per dense iteration), which is what OOM-killed
   τ-driven episodes.
3. **`cts/mcts/cts_episode.py` — parent context encoded once per leaf
   expansion** and passed to all W sibling `transition()` calls (new optional
   `context=` parameter, default `None` keeps the old per-branch encoding
   for every other caller). On the real Gemma backbone `encode_context` is a
   full-model forward pass, so this removes (W−1)/W ≈ 67 % of the episode's
   context-encoding cost on GPU as well.
4. **Terminal best-node selection reuses the critic values already computed
   at expansion** (`TreeNode.critic_value`); the critic is deterministic on
   identical input, so this only removes the duplicate O(tree) forward passes
   + per-node GPU syncs at episode end.
5. **Micro:** MAC LUT (`lut_mac.json`) cached at module level instead of a
   disk read per transition; the 19-element flops reduction uses one
   `.tolist()` host sync instead of 19 `.item()` syncs (bit-identical Python
   float accumulation).

Items 2–5 also cut GPU-path latency (fewer full-model forwards, fewer host
syncs); item 1 is the dominant term for the dense/mock path.

## 4. Test-suite status

- Pre-optimization baseline: 548 passed, 4 failed, 4 skipped.
  - `test_qwen_adapter_cpu` — needs `transformers` (not installed in this env).
  - `test_run_sweep_K_...` — needs `data/aime/test.jsonl` (dataset not
    checked in; fetched by `scripts/download_all_benchmarks.py`).
  - `test_cts_full_episode_returns_result_within_budget` — 20 s wall-clock
    flake under concurrent load; passes alone, and robustly fixed by this
    optimization.
  - `test_faq_section_refs_subset_of_reproducibility` — pre-existing doc
    drift (`REVIEWER_FAQ.md` cited `LIMITATIONS.md` "§15", which the
    alignment test reads as a paper-section family). Fixed on this branch.
- Post-optimization: **550 passed, 2 failed, 4 skipped** — only the two
  environment-dependent failures remain (missing `transformers` /
  missing AIME dataset); the wall-clock flake and the doc-alignment failure
  are fixed. Full suite runtime dropped from 144 s to 31 s on the same
  machine (the episode integration tests dominate it).

## 5. Reproduce

```bash
# controlled workload (baseline ~17-27 s, optimized <1 s)
python scripts/bench_episode_perf.py --episodes 1 --K 64 --d 64 --sims 2

# full eval-protocol episode (optimized ~9 s; pre-optimization code OOMs)
python scripts/bench_episode_perf.py --episodes 1 --K 64 --d 64 --tau 1e13 --wall 180

# equivalence matrix
python scripts/bench_episode_perf.py --episodes 3 --K 8 --d 32 --sims 6
python scripts/bench_episode_perf.py --episodes 3 --K 16 --d 64 --sims 4
```
