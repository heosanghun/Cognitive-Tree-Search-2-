# AIME train/test contamination screen

**Verdict: WARN contamination screen** (sub-verdict: `LEXICAL_OVERLAP_ONLY`)

Sub-verdict legend: `NO_FLAGS` = both detectors clean, `NEAR_DUPLICATE` = MinHash agreed (hard contamination hit), `LEXICAL_OVERLAP_ONLY` = BM25 above threshold but MinHash clean (typically topical vocabulary overlap that needs human review).

- train file: `<repo_root>\data\aime\train_2019_2023.jsonl` (150 rows, 0 placeholders)
- test  file: `<repo_root>\data\aime\test.jsonl` (30 rows, 0 placeholders)
- BM25 normalised score >= 0.5 flagged (top_k=5 per test item)
- MinHash Jaccard >= 0.8 flagged (num_perm=128, backend=pure-python)

## BM25 lexical-overlap detector

2 pair(s) above threshold:

| train_id | test_id | normalised BM25 |
|---|---|---|
| `aime_2019_I_11` | `row_22` | 0.5673 |
| `aime_2023_II_8` | `row_6` | 0.5272 |

### BM25 flagged pair text excerpts

#### `aime_2019_I_11` <-> `row_22` (score=0.5673)

- train: In $\triangle ABC$ , the sides have integer lengths and $AB=AC$ . Circle $\omega$ has its center at the incenter of $\triangle ABC$ . An excircle of $\triangle ABC$ is a circle in the exterior of $\triangle ABC$ that is tangent to one side of the triangle and tangent to the extensions of the other two sides. Suppose that the excircle tangent to $\overline{BC}$ is internally tangent to $\omega$ , a
- test:  Isosceles triangle $\triangle ABC$ has $AB=BC$. Let $I$ be the incenter of $\triangle ABC$. The perimeters of $\triangle ABC$ and $\triangle AIC$ are in the ratio $125:6$, and all the sides of both triangles have integer lengths. Find the minimum possible value of $AB$.

#### `aime_2023_II_8` <-> `row_6` (score=0.5272)

- train: Let $\omega = \cos\frac{2\pi}{7} + i \cdot \sin\frac{2\pi}{7},$ where $i = \sqrt{-1}.$ Find the value of the product $\prod_{k=0}^6 \left(\omega^{3k} + \omega^k + 1\right).$
- test:  Find the number of functions $\pi$ mapping the set $A=\{1,2,3,4,5,6\}$ onto $A$ such that for every $a\in A$, $\pi(\pi(\pi(\pi(\pi(\pi(a))))))=a$.

## MinHash near-duplicate detector

No MinHash pair met the Jaccard threshold. PASS for this detector.

## Top-1 BM25 score distribution

- N = 30 test items
- min  = 0.1296
- mean = 0.3276
- median = 0.3130
- p95 = 0.5025
- max  = 0.5673
