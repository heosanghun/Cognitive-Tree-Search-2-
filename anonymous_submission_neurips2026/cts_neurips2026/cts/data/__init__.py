"""Data utilities (paper §7.1 train/test contamination screening).

This sub-package holds helpers that operate on the JSONL benchmark dumps in
``data/`` rather than on tensors or models. Today it only exposes the
contamination-screen utilities used to certify that no AIME 2019-2023 train
prompt is a near-duplicate of an AIME 2024-2026 test problem.
"""

from cts.data.contamination_screen import (
    bm25_overlap,
    minhash_jaccard,
    screen_aime_train_test,
    tokenize,
)

__all__ = [
    "bm25_overlap",
    "minhash_jaccard",
    "screen_aime_train_test",
    "tokenize",
]
