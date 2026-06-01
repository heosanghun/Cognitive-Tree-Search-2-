"""Train/test contamination screen for the AIME benchmark (paper §7.1).

The paper claims AIME 2026 (and the broader AIME 2024-2026 family) is the
held-out evaluation set, while Stage 2 PPO is trained on AIME 2019-2023.
Reviewers are entitled to ask: *what if a 2026 problem leaked into the
2019-2023 pool?* This module answers that question with two complementary
detectors that agree iff a true near-duplicate exists.

* :func:`bm25_overlap` -- BM25 lexical-overlap score, normalised so an
  exact duplicate is ~1.0 and unrelated text is ~0.0. Catches
  vocabulary-level leaks and paraphrases that share rare tokens (numbers,
  named entities, LaTeX commands).
* :func:`minhash_jaccard` -- MinHash signature Jaccard estimate over word
  3-grams, threshold-flagged for near-duplicate detection. Catches
  whole-sentence reordering that BM25 might rank lower than the threshold.
* :func:`screen_aime_train_test` -- top-level driver that reads two JSONLs,
  runs both detectors, and writes a Markdown verdict to disk.

Dependencies are limited to the standard library plus :mod:`numpy` (already
a hard repo dependency for the rest of the codebase). :mod:`datasketch`
is preferred when available but a deterministic pure-Python MurmurHash-style
fallback is wired in so the screen runs in any reviewer environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --- tokenisation ---------------------------------------------------------

# AIME problem texts are heavy on LaTeX (``$...$``), Greek letters and
# numerals. We strip the LaTeX delimiters but KEEP the inner symbols, then
# break on any non-word character. Lower-casing is conservative because
# AIME variable names (``A`` vs ``a``) sometimes carry meaning, but for a
# contamination screen we explicitly prefer recall over identity.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-case word-token split that survives LaTeX-heavy AIME prose.

    Tokens are runs of ``[A-Za-z0-9]``; everything else (LaTeX delimiters,
    backslash commands, punctuation, whitespace) is treated as a separator.
    Empty / one-character tokens are dropped to suppress LaTeX command
    fragments such as the leading ``\\`` of ``\\frac``.
    """
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


# --- BM25 -----------------------------------------------------------------


@dataclass
class _BM25Index:
    """Internal Okapi BM25 index over a corpus of pre-tokenised documents."""

    doc_freq: dict
    doc_len: List[int]
    avgdl: float
    n_docs: int
    docs: List[List[str]]
    k1: float = 1.5
    b: float = 0.75

    def idf(self, term: str) -> float:
        # Robertson-Walker BM25 IDF; clamp at 0 so common terms don't go
        # negative and pull the score below zero.
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0))

    def score(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        if not query_tokens:
            return 0.0
        d = self.docs[doc_idx]
        if not d:
            return 0.0
        dlen = self.doc_len[doc_idx]
        # Pre-count query term frequencies in the doc once per doc
        from collections import Counter
        d_tf = Counter(d)
        score = 0.0
        for t in query_tokens:
            tf = d_tf.get(t, 0)
            if tf == 0:
                continue
            idf = self.idf(t)
            denom = tf + self.k1 * (1.0 - self.b + self.b * dlen / self.avgdl)
            score += idf * tf * (self.k1 + 1.0) / denom
        return score


def _build_bm25_index(docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> _BM25Index:
    n = len(docs_tokens)
    doc_freq: dict = {}
    for d in docs_tokens:
        for t in set(d):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    doc_len = [len(d) for d in docs_tokens]
    avgdl = (sum(doc_len) / n) if n > 0 else 0.0
    return _BM25Index(
        doc_freq=doc_freq,
        doc_len=doc_len,
        avgdl=avgdl,
        n_docs=n,
        docs=docs_tokens,
        k1=k1,
        b=b,
    )


def bm25_overlap(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[int, int, float]]:
    """Return BM25 lexical-overlap scores for every test item.

    The corpus is ``train_texts`` (the Stage 2 PPO pool); each test item is
    used as a query. Scores are normalised by the test item's BM25 score
    against an idealised "perfect duplicate" of itself in the same corpus,
    so an exact-duplicate train item returns ~1.0 and unrelated text
    returns ~0.0.

    Parameters
    ----------
    train_texts:
        Reference corpus (typically the held-in train pool).
    test_texts:
        Query texts (typically the held-out evaluation set).
    top_k:
        Number of top-scoring train matches reported per test item.

    Returns
    -------
    list of (train_idx, test_idx, normalised_score)
        Sorted by ``test_idx`` ascending then ``normalised_score`` descending.
        Every test item contributes exactly ``min(top_k, len(train_texts))``
        entries (zero-score entries included for fully unrelated tests so
        callers can do uniform aggregation).
    """
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    if len(train_texts) == 0:
        return []
    train_tok = [tokenize(t) for t in train_texts]
    test_tok = [tokenize(t) for t in test_texts]
    # Build a SHARED corpus so IDF reflects the union of train + test;
    # this is the standard contamination-screen convention because using
    # train-only IDF would under-weight test-specific rare tokens.
    union = train_tok + test_tok
    index = _build_bm25_index(union)
    out: List[Tuple[int, int, float]] = []
    for ti, q in enumerate(test_tok):
        # Self-score = score of the query against itself in the same corpus.
        # The query lives at position len(train_tok) + ti in `union`.
        self_idx = len(train_tok) + ti
        self_score = index.score(q, self_idx)
        denom = self_score if self_score > 1e-12 else 1.0
        scored = []
        for tr_i in range(len(train_tok)):
            s = index.score(q, tr_i)
            scored.append((tr_i, ti, min(s / denom, 1.0)))
        scored.sort(key=lambda x: x[2], reverse=True)
        out.extend(scored[:top_k])
    return out


# --- MinHash --------------------------------------------------------------


_MERSENNE_PRIME = (1 << 61) - 1
"""A Mersenne prime used as the modulus of the MinHash universal-hash family
when the optional :mod:`datasketch` dependency is missing."""


def _stable_token_hash(token: str) -> int:
    """Deterministic 64-bit hash of a token (cross-Python-version stable).

    Python's built-in :func:`hash` is deliberately randomised per process
    (``PYTHONHASHSEED``) so we cannot use it for a reproducible MinHash. We
    hash the UTF-8 bytes with MD5 (collision probability is irrelevant for
    a 64-bit truncation in this regime) and treat the first 8 bytes as a
    big-endian 64-bit integer.
    """
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _shingles(tokens: Sequence[str], n: int = 3) -> List[str]:
    """Return overlapping word ``n``-grams of the input token list.

    For documents shorter than ``n``, falls back to single tokens so the
    MinHash signature is still defined (otherwise a one-word doc would
    have no shingles and trivially Jaccard-match any other one-word doc).
    """
    if len(tokens) >= n:
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return list(tokens)


def _minhash_signature(
    shingles: Sequence[str],
    coeffs_a: np.ndarray,
    coeffs_b: np.ndarray,
    prime: int = _MERSENNE_PRIME,
) -> np.ndarray:
    """Pure-numpy MinHash signature using a (a*x + b) mod p universal family.

    Returns a vector of length ``len(coeffs_a)``; each element is the
    minimum over all shingles of one permutation. Empty shingle list
    yields a sentinel signature of all-``prime`` so its Jaccard estimate
    against any other signature is exactly 0.0.
    """
    num_perm = len(coeffs_a)
    if not shingles:
        return np.full(num_perm, prime, dtype=np.uint64)
    base = np.array([_stable_token_hash(s) for s in shingles], dtype=np.uint64)
    # Broadcast: (S, 1) * (1, P) + (1, P)  -> (S, P), then take min over S.
    # We work in Python int space briefly to avoid overflow in the
    # multiplication, then cast back to uint64 for the min reduction.
    sigs = np.empty(num_perm, dtype=np.uint64)
    for p in range(num_perm):
        a = int(coeffs_a[p])
        b = int(coeffs_b[p])
        h = (a * base.astype(object) + b) % prime
        sigs[p] = int(h.min())
    return sigs


def _datasketch_available() -> bool:
    try:
        import datasketch  # noqa: F401
    except Exception:
        return False
    return True


def minhash_jaccard(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    seed: int = 1729,
) -> List[Tuple[int, int, float]]:
    """Flag (train_idx, test_idx, jaccard) pairs whose MinHash Jaccard >= ``threshold``.

    Uses :mod:`datasketch.MinHashLSH` when available, otherwise falls back
    to a deterministic pure-numpy implementation (MD5-based universal
    hashing modulo a 61-bit Mersenne prime). The fallback is O(N_train *
    N_test) which is fine for AIME-scale corpora (150 x 30 = 4500 pairs).

    Parameters
    ----------
    train_texts, test_texts:
        Documents to compare. Tokenisation matches :func:`bm25_overlap`.
    threshold:
        Inclusive Jaccard threshold below which a pair is *not* reported.
    num_perm:
        Number of MinHash permutations. 128 is the datasketch default and
        gives ~0.07 standard error for true Jaccard near 0.5.
    seed:
        Seed for the universal-hash coefficients; set to make the screen
        bit-for-bit reproducible across runs.
    """
    if num_perm <= 0:
        raise ValueError("num_perm must be >= 1")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if not train_texts or not test_texts:
        return []

    train_shingles = [_shingles(tokenize(t)) for t in train_texts]
    test_shingles = [_shingles(tokenize(t)) for t in test_texts]

    if _datasketch_available():
        from datasketch import MinHash, MinHashLSH

        def _ds_minhash(toks: Iterable[str]) -> "MinHash":
            mh = MinHash(num_perm=num_perm, seed=seed)
            for t in toks:
                mh.update(t.encode("utf-8"))
            return mh

        train_mhs = [_ds_minhash(s) for s in train_shingles]
        test_mhs = [_ds_minhash(s) for s in test_shingles]
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for i, mh in enumerate(train_mhs):
            lsh.insert(f"train::{i}", mh)
        flagged: List[Tuple[int, int, float]] = []
        for ti, mh in enumerate(test_mhs):
            for key in lsh.query(mh):
                tr_i = int(key.split("::", 1)[1])
                jacc = float(train_mhs[tr_i].jaccard(mh))
                if jacc >= threshold:
                    flagged.append((tr_i, ti, jacc))
        flagged.sort(key=lambda x: (-x[2], x[1], x[0]))
        return flagged

    # Pure-Python / numpy fallback
    rng = np.random.default_rng(seed)
    coeffs_a = rng.integers(low=1, high=_MERSENNE_PRIME, size=num_perm, dtype=np.int64)
    coeffs_b = rng.integers(low=0, high=_MERSENNE_PRIME, size=num_perm, dtype=np.int64)
    train_sigs = [_minhash_signature(s, coeffs_a, coeffs_b) for s in train_shingles]
    test_sigs = [_minhash_signature(s, coeffs_a, coeffs_b) for s in test_shingles]
    flagged = []
    for ti, t_sig in enumerate(test_sigs):
        for tr_i, tr_sig in enumerate(train_sigs):
            jacc = float(np.mean(t_sig == tr_sig))
            if jacc >= threshold:
                flagged.append((tr_i, ti, jacc))
    flagged.sort(key=lambda x: (-x[2], x[1], x[0]))
    return flagged


# --- driver ---------------------------------------------------------------


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _row_id(row: dict, fallback_idx: int) -> str:
    for key in ("id", "task_id", "unique_id"):
        if key in row and row[key]:
            return str(row[key])
    return f"row_{fallback_idx}"


def _row_text(row: dict) -> str:
    for key in ("problem", "question", "input", "prompt"):
        if key in row and row[key]:
            return str(row[key])
    return ""


def screen_aime_train_test(
    train_jsonl: Path,
    test_jsonl: Path,
    output_md: Path,
    bm25_flag_threshold: float = 0.5,
    bm25_top_k: int = 5,
    minhash_threshold: float = 0.8,
    num_perm: int = 128,
) -> dict:
    """Top-level driver: load both JSONLs, run both detectors, write a report.

    Returns
    -------
    dict with keys
        ``verdict`` (``"PASS"`` / ``"WARN"`` / ``"FAIL"``),
            * ``FAIL`` -- MinHash near-duplicate hit (binding gate).
            * ``WARN`` -- BM25 lexical-overlap hit, no near-duplicate.
            * ``PASS`` -- both detectors clean.
        ``sub_verdict`` (``"NO_FLAGS"`` / ``"LEXICAL_OVERLAP_ONLY"`` / ``"NEAR_DUPLICATE"``),
        ``n_train``, ``n_test``,
        ``bm25_flagged`` (list of (train_id, test_id, score)),
        ``minhash_flagged`` (list of (train_id, test_id, jaccard)),
        ``report_path`` (the Markdown file written),
        and the raw threshold/top_k parameters for audit.

    The report is a self-contained Markdown document with a verdict header,
    the parameters used, the flagged pairs (or an empty table), and the
    text excerpts of every flagged pair so a reviewer can confirm each
    hit without re-running the screen.
    """
    train_jsonl = Path(train_jsonl)
    test_jsonl = Path(test_jsonl)
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    train_rows = _load_jsonl(train_jsonl)
    test_rows = _load_jsonl(test_jsonl)
    train_texts = [_row_text(r) for r in train_rows]
    test_texts = [_row_text(r) for r in test_rows]
    train_ids = [_row_id(r, i) for i, r in enumerate(train_rows)]
    test_ids = [_row_id(r, i) for i, r in enumerate(test_rows)]

    bm25_pairs = bm25_overlap(train_texts, test_texts, top_k=bm25_top_k)
    bm25_flagged = [(i, j, s) for (i, j, s) in bm25_pairs if s >= bm25_flag_threshold]
    minhash_flagged = minhash_jaccard(
        train_texts, test_texts, threshold=minhash_threshold, num_perm=num_perm
    )

    # Verdict policy (refined post-D2 review):
    #   FAIL  -- MinHash near-duplicate hit.  This is the BINDING gate: a true
    #            paraphrase / near-dup of a held-out test item appearing in
    #            the train set invalidates the AIME headline number.  CI must
    #            block on this.
    #   WARN  -- BM25 lexical-overlap hit but no MinHash near-dup.  This is
    #            topical / vocabulary overlap (the same competition vocabulary
    #            crops up across years).  Surface to the reviewer for manual
    #            inspection in the rendered Markdown, but do NOT block CI.
    #   PASS  -- Both detectors clean.
    if minhash_flagged:
        verdict = "FAIL"
        sub_verdict = "NEAR_DUPLICATE"
    elif bm25_flagged:
        verdict = "WARN"
        sub_verdict = "LEXICAL_OVERLAP_ONLY"
    else:
        verdict = "PASS"
        sub_verdict = "NO_FLAGS"

    placeholder_rows = sum(1 for r in train_rows if r.get("source") == "placeholder")
    test_placeholder_rows = sum(1 for r in test_rows if r.get("source") == "placeholder")

    lines: List[str] = []
    lines.append(f"# AIME train/test contamination screen")
    lines.append("")
    lines.append(f"**Verdict: {verdict} contamination screen** (sub-verdict: `{sub_verdict}`)")
    lines.append("")
    lines.append(
        "Sub-verdict legend: `NO_FLAGS` = both detectors clean, "
        "`NEAR_DUPLICATE` = MinHash agreed (hard contamination hit), "
        "`LEXICAL_OVERLAP_ONLY` = BM25 above threshold but MinHash clean "
        "(typically topical vocabulary overlap that needs human review)."
    )
    lines.append("")
    lines.append(f"- train file: `{train_jsonl}` ({len(train_rows)} rows, {placeholder_rows} placeholders)")
    lines.append(f"- test  file: `{test_jsonl}` ({len(test_rows)} rows, {test_placeholder_rows} placeholders)")
    lines.append(f"- BM25 normalised score >= {bm25_flag_threshold} flagged "
                 f"(top_k={bm25_top_k} per test item)")
    lines.append(f"- MinHash Jaccard >= {minhash_threshold} flagged "
                 f"(num_perm={num_perm}, "
                 f"backend={'datasketch' if _datasketch_available() else 'pure-python'})")
    lines.append("")

    lines.append("## BM25 lexical-overlap detector")
    lines.append("")
    if not bm25_flagged:
        lines.append("No BM25 pair scored above the flag threshold. PASS for this detector.")
    else:
        lines.append(f"{len(bm25_flagged)} pair(s) above threshold:")
        lines.append("")
        lines.append("| train_id | test_id | normalised BM25 |")
        lines.append("|---|---|---|")
        for tr_i, te_i, score in sorted(bm25_flagged, key=lambda x: -x[2]):
            lines.append(
                f"| `{train_ids[tr_i]}` | `{test_ids[te_i]}` | {score:.4f} |"
            )
        lines.append("")
        lines.append("### BM25 flagged pair text excerpts")
        for tr_i, te_i, score in sorted(bm25_flagged, key=lambda x: -x[2]):
            lines.append("")
            lines.append(f"#### `{train_ids[tr_i]}` <-> `{test_ids[te_i]}` (score={score:.4f})")
            lines.append("")
            lines.append("- train: " + train_texts[tr_i][:400].replace("\n", " "))
            lines.append("- test:  " + test_texts[te_i][:400].replace("\n", " "))
    lines.append("")

    lines.append("## MinHash near-duplicate detector")
    lines.append("")
    if not minhash_flagged:
        lines.append("No MinHash pair met the Jaccard threshold. PASS for this detector.")
    else:
        lines.append(f"{len(minhash_flagged)} pair(s) above threshold:")
        lines.append("")
        lines.append("| train_id | test_id | Jaccard |")
        lines.append("|---|---|---|")
        for tr_i, te_i, jacc in minhash_flagged:
            lines.append(
                f"| `{train_ids[tr_i]}` | `{test_ids[te_i]}` | {jacc:.4f} |"
            )
        lines.append("")
        lines.append("### MinHash flagged pair text excerpts")
        for tr_i, te_i, jacc in minhash_flagged:
            lines.append("")
            lines.append(f"#### `{train_ids[tr_i]}` <-> `{test_ids[te_i]}` (Jaccard={jacc:.4f})")
            lines.append("")
            lines.append("- train: " + train_texts[tr_i][:400].replace("\n", " "))
            lines.append("- test:  " + test_texts[te_i][:400].replace("\n", " "))
    lines.append("")

    lines.append("## Top-1 BM25 score distribution")
    lines.append("")
    if bm25_pairs:
        # Pick the highest BM25 score per test item for the histogram
        per_test_top: dict = {}
        for tr_i, te_i, score in bm25_pairs:
            if score > per_test_top.get(te_i, -1.0):
                per_test_top[te_i] = score
        scores = sorted(per_test_top.values())
        if scores:
            arr = np.asarray(scores)
            lines.append(f"- N = {len(scores)} test items")
            lines.append(f"- min  = {arr.min():.4f}")
            lines.append(f"- mean = {arr.mean():.4f}")
            lines.append(f"- median = {float(np.median(arr)):.4f}")
            lines.append(f"- p95 = {float(np.quantile(arr, 0.95)):.4f}")
            lines.append(f"- max  = {arr.max():.4f}")
    else:
        lines.append("(no pairs scored)")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "verdict": verdict,
        "sub_verdict": sub_verdict,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "bm25_flagged": [
            (train_ids[i], test_ids[j], float(s)) for (i, j, s) in bm25_flagged
        ],
        "minhash_flagged": [
            (train_ids[i], test_ids[j], float(s)) for (i, j, s) in minhash_flagged
        ],
        "report_path": str(output_md),
        "bm25_flag_threshold": bm25_flag_threshold,
        "bm25_top_k": bm25_top_k,
        "minhash_threshold": minhash_threshold,
        "num_perm": num_perm,
    }
