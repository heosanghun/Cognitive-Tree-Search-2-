# Paper vs Local &mdash; Table 2 cross-reference index

> **Status banner**: this document is a cross-reference index linking
> paper Table 2's reported headline numbers to their reviewer-facing
> reproduction evidence inside this codebase. The authoritative
> headline accuracy figures themselves live in the paper PDF
> (Table 2); the per-component code coverage map lives in
> [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) &sect;5; reviewer-facing
> FAQ entries live in [`REVIEWER_FAQ.md`](../../REVIEWER_FAQ.md)
> (Q1-Q18); open methodological caveats are consolidated in
> [`LIMITATIONS.md`](../../LIMITATIONS.md).

## Cross-reference index

| Paper element | Reviewer-facing entry inside this repository |
|:---|:---|
| §3-§7 method ↔ code anchors (Table 2 row computation) | [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) §5 (25 rows) |
| Algorithm 1 — CTS Full Episode Loop | [`cts/mcts/cts_episode.py`](../../cts/mcts/cts_episode.py) `cts_full_episode()` |
| §4.1 — Meta-Policy πφ + Neuro-Critic Vψ (4-d ν vector, Eq. 1) | [`cts/policy/meta_policy.py`](../../cts/policy/meta_policy.py), [`cts/critic/neuro_critic.py`](../../cts/critic/neuro_critic.py) |
| §4.2 — KV-Cache-Free DEQ Transition | [`cts/deq/transition.py`](../../cts/deq/transition.py), [`cts/deq/broyden_forward.py`](../../cts/deq/broyden_forward.py) |
| §4.3 — Wproj soft-prompt decoding + FAISS-IVF-PQ Latent Context | [`cts/backbone/gemma_adapter.py`](../../cts/backbone/gemma_adapter.py) `decode_from_z_star()`, [`cts/latent/faiss_context.py`](../../cts/latent/faiss_context.py) |
| §5.3 Eq. 3 — Sparse Top-k Routing (CPU reference + Triton fused kernel) | [`cts/routing/sparse_moe_ref.py`](../../cts/routing/sparse_moe_ref.py), [`cts/routing/sparse_moe_triton.py`](../../cts/routing/sparse_moe_triton.py) |
| §6 — Stage 1 (DEQ warm-up, IFT + 0.1·LCE) + Stage 2 (PPO + GAE) | [`cts/train/stage1_warmup.py`](../../cts/train/stage1_warmup.py), [`cts/train/stage2_ppo_train.py`](../../cts/train/stage2_ppo_train.py), [`cts/train/ppo_core.py`](../../cts/train/ppo_core.py) |
| §7.1 — Statistical protocol (bootstrap CI + Wilcoxon + Bonferroni) | [`cts/eval/statistics.py`](../../cts/eval/statistics.py) |
| §7.7 — Hybrid KV-Assisted Mode (decision-plumbed) | [`cts/mcts/hybrid_kv.py`](../../cts/mcts/hybrid_kv.py) |
| Reviewer Quick Start audit (52/52 PASS, ~0.5 s, no torch) | [`scripts/_reviewer_local_audit.py`](../../scripts/_reviewer_local_audit.py) |
| Anonymous submission ZIP | [`anonymous_submission_neurips2026.zip`](../../anonymous_submission_neurips2026.zip) |

## Table 2 baseline implementation status (per-method)

This is the per-method status referenced from
[`LIMITATIONS.md`](../../LIMITATIONS.md) §5.

> **Reading the table.** Of the paper's 14 Table 2 rows,
> **12 dispatcher paths are wired** through `_run_cts_on_problems`
> ([`scripts/run_cts_eval_full.py`](../../scripts/run_cts_eval_full.py))
> and verified by
> [`tests/test_baseline_dispatchers.py`](../../tests/test_baseline_dispatchers.py)
> &mdash; **10 paper-faithful** plus **2 proxies with explicitly
> documented gaps** (rows 10-11 below). The remaining **2 rows
> (paper Table 2 rows 6-7: COCONUT and Recurrent Depth) are openly
> disclosed as not-yet-implemented**, reserved for the camera-ready
> window. The aggregated view in LIMITATIONS §5 ("**4 of 14 baselines
> not paper-faithful**") collapses the 2 missing + 2 proxy rows into
> a single bucket; this table preserves the finer-grained 10 / 2 / 2
> split so reviewers can distinguish a *missing dispatcher entry*
> from a *proxy entry with a documented gap*.

| Paper Table 2 row | Method | Status | Implementation anchor / disclosure |
|:---:|:---|:---:|:---|
| 1 | Greedy (standard, chat-template) | ✅ implemented | `_run_cts_on_problems::method=="greedy"` (paper-faithful) |
| 2 | Think-OFF Greedy | ✅ implemented | `_run_cts_on_problems::method=="think_off_greedy"` (paper-faithful) |
| 3 | Native Think | ✅ implemented | `_run_cts_on_problems::method=="native_think"` (paper-faithful) |
| 4 | FT-NT (native-think with Stage 1 LoRA) | ✅ implemented | `_run_cts_on_problems::method=="ft_nt"` (LoRA hot-swap deferred; banner-disclosed) |
| 5 | Self-Consistency @ K=14 | ✅ implemented | `_run_cts_on_problems::method=="sc_14"` (paper-faithful: T=0.7, majority vote) |
| 6 | **COCONUT** (Gemma-4-E4B reproduction) | ❌ **missing** | no dispatcher entry; reserved for camera-ready (LIMITATIONS §5) |
| 7 | **Recurrent Depth** (Gemma-4-E4B) | ❌ **missing** | no dispatcher entry; reserved for camera-ready (LIMITATIONS §5) |
| 8 | MCTS Early-Stop | ✅ implemented | `_run_cts_on_problems::method=="mcts_early_stop"` (30% τ-cap, 60 s wall-clock) |
| 9 | EXPL-MCTS-PPO | ✅ implemented | `_run_cts_on_problems::method=="expl_mcts_ppo"` (depth cap 15, no FAISS context) |
| 10 | **BoN@13** (Critic-best argmax) | ⚠️ **proxy** (paper-faithful: no) | dispatcher entry exists (`method=="bon_13"`) but uses *longest-well-formed-chain* as a coarse proxy for V_ψ scoring; paper protocol uses Neuro-Critic V_ψ directly. Disclosed in `_run_cts_on_problems` dispatcher comment + [`CHANGELOG.md`](../../CHANGELOG.md) D1 P1 baseline-dispatcher sweep |
| 11 | **UCB1 Bandit** (20-bin ν, c=√2) | ⚠️ **proxy** (paper-faithful: no) | dispatcher entry exists (`method=="bandit_ucb1"`) but routes through `cts_full_episode` with `nu_config_mode="1nu"` (only ν_expl live, all others frozen at Stage 1 means); paper protocol uses a dedicated 20-arm UCB1 bandit module. Disclosed in `_run_cts_on_problems` dispatcher comment + [`CHANGELOG.md`](../../CHANGELOG.md) D1 P1 baseline-dispatcher sweep |
| 12 | DEQ-only (no MCTS) | ✅ implemented | `_run_cts_on_problems::method=="deq_only"` (paper-faithful) |
| 13 | CTS-2ν (2-coordinate ablation) | ✅ implemented | `_run_cts_on_problems::method=="cts_2nu"` (`nu_config_mode="2nu_fast"`) |
| 14 | CTS-4ν (full method) | ✅ implemented | `_run_cts_on_problems::method=="cts_4nu"` (`nu_config_mode="4nu"`) |

**Legend.** ✅ implemented = paper-faithful dispatcher path with
passing unit tests · ⚠️ proxy = dispatcher entry exists with an
explicit caveat short of paper-faithful · ❌ missing = no dispatcher
entry, reserved for camera-ready.

The two ❌ rows (paper Table 2 rows 6-7) are scaffolding-level gaps
intentionally deferred to the camera-ready window; the two ⚠️ rows
(paper Table 2 rows 10-11) are running today as proxies whose specific
gap is named in the table's last column. Both buckets are aggregated
under the LIMITATIONS §5 disclosure of "4 of 14 baselines not
paper-faithful" so reviewers reading either document arrive at the
same total without double-counting.

## Why the Gap?

The headline absolute Table 2 numbers depend on the paper's full
compute envelope (8&times;A100, &tau;<sub>budget</sub>=10<sup>14</sup>,
multi-host PPO budget) which a single-host single-GPU reproduction
window cannot match without proportional compute scaling. Within that
constraint, the *relative ordering* of methods, the per-Node O(1) VRAM
signature (paper Table 1, structurally guaranteed by the KV-cache-free
DEQ transition), and the &nu;-control adaptive-operator mechanism
(paper Table 19) all reproduce on the local hardware class.
The full paper-headline accuracy on the paper backbone (Gemma 4 E4B)
is reserved for the camera-ready window. See
[`REVIEWER_FAQ.md`](../../REVIEWER_FAQ.md) Q4 for the per-row scaling
factor analysis, Q13 for the per-cell methodological context, and
Q15 for the single-host environment caveat.
