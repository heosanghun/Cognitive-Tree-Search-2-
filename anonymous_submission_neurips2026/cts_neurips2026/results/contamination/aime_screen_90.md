# AIME train/test contamination screen

**Verdict: WARN contamination screen** (sub-verdict: `LEXICAL_OVERLAP_ONLY`)

Sub-verdict legend: `NO_FLAGS` = both detectors clean, `NEAR_DUPLICATE` = MinHash agreed (hard contamination hit), `LEXICAL_OVERLAP_ONLY` = BM25 above threshold but MinHash clean (typically topical vocabulary overlap that needs human review).

- train file: `data\aime\train_2019_2023.jsonl` (150 rows, 0 placeholders)
- test  file: `data\aime\test_aime_90.jsonl` (90 rows, 0 placeholders)
- BM25 normalised score >= 0.5 flagged (top_k=5 per test item)
- MinHash Jaccard >= 0.8 flagged (num_perm=128, backend=pure-python)

## BM25 lexical-overlap detector

6 pair(s) above threshold:

| train_id | test_id | normalised BM25 |
|---|---|---|
| `aime_2020_I_12` | `aime_2024_I_13` | 0.6393 |
| `aime_2021_I_14` | `aime_2024_I_13` | 0.5727 |
| `aime_2019_I_11` | `aime_2026_unknown_22` | 0.5623 |
| `aime_2021_II_2` | `aime_2025_I_2` | 0.5430 |
| `aime_2022_II_15` | `aime_2025_II_6` | 0.5172 |
| `aime_2023_II_8` | `aime_2026_unknown_06` | 0.5092 |

### BM25 flagged pair text excerpts

#### `aime_2020_I_12` <-> `aime_2024_I_13` (score=0.6393)

- train: Let $n$ be the least positive integer for which $149^n-2^n$ is divisible by $3^3\cdot5^5\cdot7^7.$ Find the number of positive integer divisors of $n.$
- test:  Let $p$ be the least prime number for which there exists an integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$ . Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$ .

#### `aime_2021_I_14` <-> `aime_2024_I_13` (score=0.5727)

- train: For any positive integer $a,$ $\sigma(a)$ denotes the sum of the positive integer divisors of $a$ . Let $n$ be the least positive integer such that $\sigma(a^n)-1$ is divisible by $2021$ for all positive integers $a$ . Find the sum of the prime factors in the prime factorization of $n$ .
- test:  Let $p$ be the least prime number for which there exists an integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$ . Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$ .

#### `aime_2019_I_11` <-> `aime_2026_unknown_22` (score=0.5623)

- train: In $\triangle ABC$ , the sides have integer lengths and $AB=AC$ . Circle $\omega$ has its center at the incenter of $\triangle ABC$ . An excircle of $\triangle ABC$ is a circle in the exterior of $\triangle ABC$ that is tangent to one side of the triangle and tangent to the extensions of the other two sides. Suppose that the excircle tangent to $\overline{BC}$ is internally tangent to $\omega$ , a
- test:  Isosceles triangle $\triangle ABC$ has $AB=BC$. Let $I$ be the incenter of $\triangle ABC$. The perimeters of $\triangle ABC$ and $\triangle AIC$ are in the ratio $125:6$, and all the sides of both triangles have integer lengths. Find the minimum possible value of $AB$.

#### `aime_2021_II_2` <-> `aime_2025_I_2` (score=0.5430)

- train: Equilateral triangle $ABC$ has side length $840$ . Point $D$ lies on the same side of line $BC$ as $A$ such that $\overline{BD} \perp \overline{BC}$ . The line $\ell$ through $D$ parallel to line $BC$ intersects sides $\overline{AB}$ and $\overline{AC}$ at points $E$ and $F$ , respectively. Point $G$ lies on $\ell$ such that $F$ is between $E$ and $G$ , $\triangle AFG$ is isosceles, and the ratio 
- test:  On $\triangle ABC$ points $A$ , $D$ , $E$ , and $B$ lie in that order on side $\overline{AB}$ with $AD = 4$ , $DE = 16$ , and $EB = 8$ . Points $A$ , $F$ , $G$ , and $C$ lie in that order on side $\overline{AC}$ with $AF = 13$ , $FG = 52$ , and $GC = 26$ . Let $M$ be the reflection of $D$ through $F$ , and let $N$ be the reflection of $G$ through $E$ . Quadrilateral $DEGF$ has area $288$ . Find th

#### `aime_2022_II_15` <-> `aime_2025_II_6` (score=0.5172)

- train: Two externally tangent circles $\omega_1$ and $\omega_2$ have centers $O_1$ and $O_2$ , respectively. A third circle $\Omega$ passing through $O_1$ and $O_2$ intersects $\omega_1$ at $B$ and $C$ and $\omega_2$ at $A$ and $D$ , as shown. Suppose that $AB = 2$ , $O_1O_2 = 15$ , $CD = 16$ , and $ABO_1CDO_2$ is a convex hexagon. Find the area of this hexagon. [asy] import geometry; size(10cm); point O
- test:  Circle $\omega_1$ with radius $6$ centered at point $A$ is internally tangent at point $B$ to circle $\omega_2$ with radius $15$ . Points $C$ and $D$ lie on $\omega_2$ such that $\overline{BC}$ is a diameter of $\omega_2$ and ${\overline{BC} \perp \overline{AD}}$ . The rectangle $EFGH$ is inscribed in $\omega_1$ such that $\overline{EF} \perp \overline{BC}$ , $C$ is closer to $\overline{GH}$ than 

#### `aime_2023_II_8` <-> `aime_2026_unknown_06` (score=0.5092)

- train: Let $\omega = \cos\frac{2\pi}{7} + i \cdot \sin\frac{2\pi}{7},$ where $i = \sqrt{-1}.$ Find the value of the product $\prod_{k=0}^6 \left(\omega^{3k} + \omega^k + 1\right).$
- test:  Find the number of functions $\pi$ mapping the set $A=\{1,2,3,4,5,6\}$ onto $A$ such that for every $a\in A$, $\pi(\pi(\pi(\pi(\pi(\pi(a))))))=a$.

## MinHash near-duplicate detector

No MinHash pair met the Jaccard threshold. PASS for this detector.

## Top-1 BM25 score distribution

- N = 90 test items
- min  = 0.0802
- mean = 0.3258
- median = 0.3262
- p95 = 0.5024
- max  = 0.6393
