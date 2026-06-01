"""Garbage-prediction detection for math benchmarks (Q14 Fix B).

This module factors out the predicate that the dispatcher in
``scripts/run_cts_eval_full.py`` uses to decide whether a CTS / DEQ
prediction on a math benchmark looks like the AIME garbage symptom
(``Cultura``, ``LinearLayout``, ``?????``) and should be replaced
by the greedy fallback path.

It lives here (vs. inline in ``run_cts_eval_full.py``) for two
reasons:

1. The predicate is now unit-testable in <10 ms with no torch
   dependency (see ``tests/test_dispatcher_fallback_mock.py``).
2. The dispatcher can import the helper instead of duplicating
   the regex / benchmark-membership logic in two separate places
   (CTS path + DEQ-only path).

This is a *pure function*: no side effects, no torch.
"""

from __future__ import annotations

import re

# Math benchmarks where the gold answer is always numeric (integer
# or signed real). Any non-numeric prediction on these benchmarks
# is by definition wrong, so it is safe to treat as garbage and
# trigger the greedy fallback. Adding a benchmark here means the
# dispatcher will fall back on it whenever the CTS / DEQ
# prediction is non-numeric.
MATH_BENCHMARKS: frozenset[str] = frozenset({
    "math500",
    "gsm8k",
    "aime",
    "aime_90",
})

_NUMERIC_PREFIX = re.compile(r"^-?\d")


def is_math_benchmark(benchmark: str | None) -> bool:
    """Whether the given benchmark expects a numeric answer."""
    return benchmark in MATH_BENCHMARKS


def is_garbage_math(benchmark: str | None, prediction: str | None) -> bool:
    """Decide whether ``prediction`` on ``benchmark`` looks like
    the AIME garbage symptom (Q14) and should trigger the greedy
    fallback.

    Returns True iff:

    - the benchmark expects a numeric answer (math500 / gsm8k /
      aime / aime_90), AND
    - the prediction is not None, AND
    - the prediction does not start with an optional minus sign
      followed by a digit (i.e. it is non-numeric).

    None and empty-string predictions on math benchmarks are
    *also* considered garbage, so the dispatcher's existing
    ``not pred_raw or len(pred_raw.strip()) < 3`` check is
    redundant with this helper for those cases. We keep both
    checks in the dispatcher for defence-in-depth.

    Examples (math benchmarks):

        >>> is_garbage_math("aime", "Cultura")
        True
        >>> is_garbage_math("gsm8k", "LinearLayout")
        True
        >>> is_garbage_math("math500", "?????")
        True
        >>> is_garbage_math("aime", "47")
        False
        >>> is_garbage_math("math500", "-12.5")
        False
        >>> is_garbage_math("aime_90", "")
        True
        >>> is_garbage_math("aime", None)
        False

    Examples (non-math benchmarks - never garbage):

        >>> is_garbage_math("humaneval", "Cultura")
        False
        >>> is_garbage_math("arc", "B")
        False
        >>> is_garbage_math(None, "Cultura")
        False
    """
    if not is_math_benchmark(benchmark):
        return False
    if prediction is None:
        return False
    return _NUMERIC_PREFIX.match(prediction) is None
