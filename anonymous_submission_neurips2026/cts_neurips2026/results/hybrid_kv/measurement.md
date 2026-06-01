# Hybrid-KV (paper §7.7) decision-overhead measurement

> **DISCLOSURE — read this BEFORE any number below.**
>
> **KV-reuse hit path NOT YET measured.** The paper's −21 % wall-clock figure
> (§7.7) requires backbone-level `past_key_values` serialization that is not yet
> plumbed into `GemmaCTSBackbone`. This report measures only what the local
> pipeline can honestly observe today: (a) the decision overhead of consulting
> `HybridKVManager` on every leaf, and (b) the cached-node statistics surfaced by
> `HybridKVManager.report()`. The −21 % figure remains the **paper's reference
> number**, not a measured local result.

This report is the *honest* counterpart to the README's Implementation Status row that flags Hybrid-KV as `⚠️ decision-plumbed; KV-reuse pending`.

What follows is what the local pipeline CAN measure today: the wall-clock cost of consulting `HybridKVManager` on every leaf (decision overhead) plus the cache statistics surfaced by `HybridKVManager.report()`. The cache-HIT fast path is documented as future work in `cts/eval/cuda_graph_skeleton.py` and the TODO block in `cts/mcts/hybrid_kv.py::HybridKVManager.__init__`.

## 1. Configuration

- seeds: 3
- problems: 4
- TOST equivalence margin: ±5.0 % of `hybrid_off` mean (α = 0.050)

## 2. Per-mode wall-clock (mean ± std)

| mode | n | wall_seconds (mean ± std) | decision_calls (mean) | cached_nodes (mean) | vram_used_gb (mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_off` | 12 | 0.0147 ± 0.0010 | 0.0 | 0.0 | 0.000000 |
| `hybrid_decision_only` | 12 | 0.0145 ± 0.0006 | 5.0 | 0.0 | 0.000000 |

_Note: `cached_nodes` and `vram_used_gb` are expected to be **0** today because the cache HIT path is not yet plumbed. Non-zero values would indicate the post-submission `past_key_values` serialization has landed._

## 3. TOST equivalence verdict (hybrid_off vs hybrid_decision_only)

- delta (absolute):    0.000734 s
- mean_diff (off − on): 0.000204 s
- p_lower:             0.004461
- p_upper:             0.057117
- p_max:               0.057117
- **equivalent at α = 0.050: False**

## 4. What this report DOES NOT claim

- The paper's **−21 % wall-clock figure (§7.7)** is the reference number, not a measured local result on this machine. Measuring it requires the cache-HIT path documented as future work in `cts/eval/cuda_graph_skeleton.py`.
- The TOST verdict above is a *decision-overhead equivalence* test, not an accuracy-equivalence test. Once the HIT path is plumbed, reviewers should re-run this scaffold against per-seed accuracy arrays to reproduce the §7.7 'accuracy unchanged (p=0.89)' claim.
