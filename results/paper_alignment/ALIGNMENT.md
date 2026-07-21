# Paper ↔ Code Alignment Audit — submitted CTS.pdf vs this repository

Date: 2026-07-21. Source of truth: the submitted NeurIPS 2026 manuscript
(`CTS.pdf`, 21 pages incl. appendices/checklist), read in full and diffed
against the code. This complements the README's self-declared mapping table:
that table was verified previously (static audit 52/52); this audit checks the
**manuscript text itself** against code behaviour and fixes what disagreed.

## 1. Mismatches found and fixed

| # | Paper says | Code did | Fix |
|---|---|---|---|
| M1 | Eq. 2: PUCT prior **P(s,a) = 1/W (uniform)**; π_φ emits only ν ∈ R⁴ | `cts_full_episode` overwrote `mcts_prior` with the meta-policy's learned softmax priors | Learned-prior overwrite removed; `TreeNode` uniform default is used. The prior head remains for the Stage-2 PPO actor (its documented role). `cts/mcts/cts_episode.py` |
| M2 | Eq. 3: `z* = Σ_{i∈Top-k} Softmax(W_g z*/ν_temp)_i · m_i(…)` — **raw softmax weights, no renormalisation** (also serves Prop. 1's Σ g_i L_i < 1) | `sparse_module_weights` renormalised the Top-k weights to sum to 1; Triton kernel did the same | Default is now the paper's un-renormalised form; `renormalize=True` kept for back-comparison. Triton kernel emits raw selected weights; ref↔Triton parity test unchanged and passing. `cts/routing/sparse_moe_ref.py`, `sparse_moe_triton.py` |
| M3 | §4.3 / Algorithm 1 line 8: FAISS retrieval **for t > 10** (strict) | Episode gated retrieval at `t >= 10` | Strict `t > 10`. `cts/mcts/cts_episode.py` |
| M4 | Algorithm 1 lines 10–11: shallow nodes use **H_t ← AncestorStack(s)** | No ancestor context at t ≤ 10 (branches saw only the leaf encoding) | Up to 3 nearest-ancestor z\* are prepended to the leaf context through the same `prepend_soft_prefix` pathway FAISS retrieval uses. `cts/mcts/cts_episode.py` |
| M5 | Algorithm 1 lines 13–17: the W solves run first (**parallel**), then `F.add(z*_w)` happens **sequentially afterwards** | `transition()` registered each z\* into FAISS *during* the branch loop — sibling w+1 could retrieve sibling w's latent within the same expansion | `transition(faiss_add=False)` + episode-level sequential adds after the branch loop (fallback nodes excluded per Appendix K). Optional `parallel_expansion=True` runs the W solves as a thread-parallel batch per line 7. `cts/deq/transition.py`, `cts/mcts/cts_episode.py` |
| M6 | Appendix H: meta-policy + Critic LUT cost **≈0.002×10¹⁴ per episode, <0.8%** of budget | The flat constant `0.002e14` was charged **per simulation** — orders of magnitude over Appendix H, distorting τ-budget halting (~25% of accumulated MAC) | Line-5 charge is now the actual forward MACs of the two 2-layer MLPs (≈ parameter count per call), keeping the per-episode controller overhead at the documented <0.8%. `cts/mcts/cts_episode.py` |
| M7 | (docstring integrity) `sparse_moe_triton.py` quoted a "§5.3" passage that does not exist in the submitted manuscript | stale draft quote | Docstring now quotes the actual §4.1 sentence ("Triton kernels achieve 25±2 ms at W=3"). |

## 2. Paper value exposed as a parameter (with rationale)

| Item | Paper | Resolution |
|---|---|---|
| Algorithm 1 line 1: **B₀ ← 0.1·I** | Root Broyden Jacobian estimate | Implemented as `broyden_fixed_point(root_b0_scale=…)` with the paper value selectable. The **default stays 1.0**: measured on the CI mock backbone (contraction γ≈0.5), H₀=10·I diverges (residual 6.0 after 40 iters vs convergence in 11 iters at identity), which would contradict the paper's own 97.3% convergence claim (Table 12). 0.1·I is near-exact only in the paper's γ≈0.92 regime; at Gemma scale this repo routes to Anderson acceleration (no B maintained — previously documented divergence). |

## 3. Documented deviations kept (not code-fixable without contradicting the paper's own protocol)

| Item | Paper | Code | Why kept |
|---|---|---|---|
| Broyden "relative tolerance ν_tol" (§4.2) | wording suggests a relative stop criterion | absolute residual ‖F(z)‖ < tol with ν_tol mapped to [10⁻⁴,10⁻²] — the same numeric range Table 19 reports | The mapped-range values match Table 19 exactly; a literal relative-to-‖z‖ criterion cannot converge for fixed points near 0. Ambiguous wording → documented, not changed. |
| Gumbel(0, ν_temp) noise in PUCT selection | Eq. 2 is a pure argmax; ν_temp is routing-only (Eq. 3) | selection adds Gumbel noise when `selection_seed` is set | Deliberate, previously documented fix for multi-seed collapse (std = 0.0) under the paper's own 5-seed protocol; removing it would break the seed-variance the paper reports. |
| Double leaf-selection + second meta-policy query per iteration | Algorithm line 3 uses ν from the previous iteration | code re-selects once with the fresh ν | Behavioural refinement noted in-code; Algorithm is ambiguous about the first iteration's ν. |
| W-batch Triton "25±2 ms" (§4.1) | GPU figure | CUDA-only kernel present; CPU falls back to reference | GPU-only claim; not measurable in this environment. |

## 4. Verification

- Solver/routing/episode test files pass after alignment; ref↔Triton parity
  and the Sherman–Morrison equivalence pin were updated in lockstep
  (`tests/test_broyden_sherman_morrison.py` derives its classic-B₀ from
  `ROOT_B0_SCALE`).
- Note on scale: with the honest Appendix-H meta-MAC accounting (M6), a
  τ-driven episode now runs the *hundreds-to-thousands* of simulations the
  paper's D≈100 configuration implies, instead of being throttled by the
  inflated controller charge. Wall-clock-capped runs (the eval protocol's
  `CTS_EVAL_EPISODE_TIMEOUT`) are unaffected in duration but explore more.
- Full-suite result recorded in CHANGELOG for this commit.
