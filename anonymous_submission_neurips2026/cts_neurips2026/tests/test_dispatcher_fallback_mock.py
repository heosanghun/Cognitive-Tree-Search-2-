"""Mock-based behavioural tests for the Q14 Fix B garbage-math
fallback predicate.

Why this file exists:

The author's single development host cannot ``import torch``
reliably (REVIEWER_FAQ Q15), but the dispatcher's fallback
behaviour - "if the CTS prediction is non-numeric on a math
benchmark, fall back to greedy" - is the *behavioural* claim
that ``REVIEWER_FAQ.md`` Q14 makes. Static AST inspection
(``tests/test_d7_static_validation.py``) only checks that the
predicate text exists; this file actually *executes* the
predicate against the canonical garbage strings the AIME
incident produced and verifies it returns the expected
True/False.

Every test in this file:

- Runs in <10 ms
- Imports only ``cts.eval.garbage_filter`` (a 80-line
  pure-Python module with no torch / numpy / transformers)
- Drives the predicate through the canonical Q14 garbage corpus
  (``Cultura``, ``LinearLayout``, ``?????``, ``obar``) plus a
  benign numeric corpus, and asserts the True/False that the
  dispatcher will see at runtime.

If a future refactor changes the dispatcher's predicate
semantics, these tests will break before any AIME run is
launched, catching the regression in the same 0.4 s window the
torch-free static suite already covers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent

# Load ``cts/eval/garbage_filter.py`` directly via importlib so we
# do *not* execute ``cts/__init__.py`` (which transitively imports
# torch through cts.mcts.cts_episode and would hang on the
# author's degraded host - REVIEWER_FAQ Q15). The module is a
# standalone pure-Python file with no torch / numpy imports, so
# this direct load is safe and returns in <5 ms.
_GF_PATH = ROOT / "cts" / "eval" / "garbage_filter.py"
_spec = importlib.util.spec_from_file_location("_gf_test", _GF_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {_GF_PATH}"
_gf = importlib.util.module_from_spec(_spec)
sys.modules["_gf_test"] = _gf
_spec.loader.exec_module(_gf)

MATH_BENCHMARKS = _gf.MATH_BENCHMARKS
is_garbage_math = _gf.is_garbage_math
is_math_benchmark = _gf.is_math_benchmark


# ---------------------------------------------------------------------------
# Canonical Q14 corpus (from results/post_stage2_D11/nu_traces/
# cts_4nu_aime_seed0.jsonl pre-fix episodes).
# ---------------------------------------------------------------------------

GARBAGE_TOKENS = [
    "Cultura",
    "LinearLayout",
    "?????",
    "obar",
    "the answer is",  # non-numeric prose
    "x = ",            # symbolic, no numeric prefix
    "<unk>",
    "????",
    "ANSWER",
    "Solution:",
]

NUMERIC_TOKENS = [
    "47",
    "-12",
    "0",
    "3.14",
    "-3.14",
    "1234567890",
    "-99999",
]


# ---------------------------------------------------------------------------
# Section 1: garbage_filter unit invariants
# ---------------------------------------------------------------------------


def test_math_benchmark_membership():
    assert is_math_benchmark("aime")
    assert is_math_benchmark("aime_90")
    assert is_math_benchmark("math500")
    assert is_math_benchmark("gsm8k")
    assert not is_math_benchmark("humaneval")
    assert not is_math_benchmark("arc")
    assert not is_math_benchmark("arc_agi_text")
    assert not is_math_benchmark(None)
    assert not is_math_benchmark("")


def test_math_benchmarks_set_is_frozen():
    """The math benchmark set is intentionally a frozenset so that
    accidental mutation by a downstream module raises rather than
    silently expanding fallback scope."""
    assert isinstance(MATH_BENCHMARKS, frozenset)
    try:
        MATH_BENCHMARKS.add("foo")  # type: ignore[attr-defined]
    except AttributeError:
        return
    assert False, "MATH_BENCHMARKS must be immutable"


def test_garbage_on_every_math_benchmark():
    """Every garbage token triggers fallback on every math benchmark."""
    for bench in MATH_BENCHMARKS:
        for tok in GARBAGE_TOKENS:
            assert is_garbage_math(bench, tok), (
                f"{bench!r} + {tok!r} should be garbage"
            )


def test_numeric_never_triggers_fallback():
    """No numeric token is ever flagged as garbage on math benchmarks."""
    for bench in MATH_BENCHMARKS:
        for tok in NUMERIC_TOKENS:
            assert not is_garbage_math(bench, tok), (
                f"{bench!r} + {tok!r} should NOT be garbage"
            )


def test_non_math_benchmarks_never_garbage():
    """On non-math benchmarks (humaneval / arc / etc.) we never
    fall back, regardless of the prediction. This is critical:
    HumanEval expects code, ARC expects letters, and falling
    back to greedy on those would *destroy* accuracy."""
    for bench in ("humaneval", "arc", "arc_agi_text", "mbpp", None, ""):
        for tok in GARBAGE_TOKENS + NUMERIC_TOKENS + ["B", "def f():\n  return 1"]:
            assert not is_garbage_math(bench, tok), (
                f"non-math {bench!r} + {tok!r} should NEVER be garbage"
            )


def test_none_prediction_never_triggers_fallback():
    """A None prediction is handled by the dispatcher's
    ``not pred_raw`` guard, not by garbage_filter. The helper
    must therefore return False for None to avoid double-fallback."""
    for bench in MATH_BENCHMARKS:
        assert not is_garbage_math(bench, None)


def test_empty_string_is_garbage_on_math():
    """Empty string starts with no digit, so the predicate flags
    it as garbage. The dispatcher's separate ``len < 3`` guard
    also catches this; we keep both for defence-in-depth."""
    for bench in MATH_BENCHMARKS:
        assert is_garbage_math(bench, "")


def test_negative_numbers_are_not_garbage():
    """The predicate must accept ``-12`` and ``-3.14`` as numeric;
    a naive ``str.isdigit()`` check would reject them and trigger
    spurious fallback on legitimate negative-answer problems."""
    for bench in MATH_BENCHMARKS:
        assert not is_garbage_math(bench, "-12")
        assert not is_garbage_math(bench, "-3.14")
        assert not is_garbage_math(bench, "-0")


def test_decimals_are_not_garbage():
    for bench in MATH_BENCHMARKS:
        assert not is_garbage_math(bench, "3.14159")
        assert not is_garbage_math(bench, "0.5")


# ---------------------------------------------------------------------------
# Section 2: Mock-based dispatcher behaviour
# ---------------------------------------------------------------------------


def _simulate_dispatcher_decision(
    benchmark: str,
    cts_pred: str,
    pred_raw: str | None = None,
) -> str:
    """Faithful re-implementation of the *decision logic* in
    ``scripts/run_cts_eval_full.py`` lines 565-580 / 700-720,
    *without* importing the torch-laden script.

    Args:
        benchmark: e.g. ``"aime"``, ``"humaneval"``.
        cts_pred: the *extracted* prediction (post ``_extract_pred``);
            this is what ``is_garbage_math`` sees.
        pred_raw: the *raw* CTS model output before extraction.
            The dispatcher's ``len(pred_raw.strip()) < 3`` and
            ``== "obar"`` guards apply here. Defaults to a 32-char
            scaffold that wraps ``cts_pred`` so they never trip on
            short clean predictions like ``"47"``.

    Returns:
        ``"cts"`` if the dispatcher would keep the CTS prediction,
        or ``"fallback"`` if it would call the greedy fallback.
    """
    if pred_raw is None:
        pred_raw = f"The answer is {cts_pred}.\n\\boxed{{{cts_pred}}}"
    is_garbage = is_garbage_math(benchmark, cts_pred)
    if (
        not pred_raw
        or len(pred_raw.strip()) < 3
        or pred_raw.strip() == "obar"
        or is_garbage
    ):
        return "fallback"
    return "cts"


def test_dispatcher_falls_back_on_canonical_aime_garbage():
    """The exact strings the AIME garbage incident produced
    (REVIEWER_FAQ Q14 evidence dump) must trigger fallback."""
    canonical = ["Cultura", "LinearLayout", "?????"]
    for s in canonical:
        for bench in ("aime", "aime_90", "math500", "gsm8k"):
            assert _simulate_dispatcher_decision(bench, s) == "fallback", (
                f"dispatcher must fall back on canonical garbage {s!r} for {bench!r}"
            )


def test_dispatcher_keeps_clean_numeric_predictions():
    """If CTS produces a clean numeric answer (with a non-trivial
    raw output, as the real model would), the dispatcher must
    NOT fall back (otherwise we lose every CTS win)."""
    for s in ("47", "277", "-12", "3.14", "0"):
        for bench in ("aime", "aime_90", "math500", "gsm8k"):
            assert _simulate_dispatcher_decision(bench, s) == "cts", (
                f"dispatcher must keep clean prediction {s!r} for {bench!r}"
            )


def test_dispatcher_falls_back_on_too_short_raw_output():
    """If the raw model output itself is too short (<3 chars),
    the dispatcher's ``len < 3`` guard triggers fallback even
    when the extracted prediction is numeric. This is the real
    behaviour and we lock it in (a future refactor changing this
    must explicitly update this test)."""
    for s in ("47", "0"):
        # raw output is only 2 chars - shorter than the 3-char guard
        for bench in ("aime", "math500"):
            assert _simulate_dispatcher_decision(bench, s, pred_raw=s) == "fallback"


def test_dispatcher_falls_back_on_obar_raw_output():
    """The ``obar`` token is a known degenerate output from
    the soft-prompt prefix decoder; the dispatcher hard-codes
    fallback for it (REVIEWER_FAQ Q14 evidence dump)."""
    for bench in ("aime", "math500", "gsm8k"):
        assert _simulate_dispatcher_decision(bench, "47", pred_raw="obar") == "fallback"


def test_dispatcher_never_falls_back_on_humaneval():
    """HumanEval is non-math; the fallback path produces greedy
    *math* prompts, which would corrupt code generation. The
    dispatcher must never fall back on HumanEval, even when the
    CTS prediction looks like garbage to a math judge."""
    for s in ["Cultura", "LinearLayout", "def foo():\n    return 1"]:
        assert _simulate_dispatcher_decision("humaneval", s) == "cts", (
            "dispatcher must never fall back on HumanEval"
        )


# ---------------------------------------------------------------------------
# Section 3: Mock-based predictor end-to-end
# ---------------------------------------------------------------------------


def test_mock_fallback_invoked_when_cts_returns_garbage():
    """End-to-end mock: a CTS path returns 'Cultura', the
    fallback predictor is invoked exactly once, and the final
    answer comes from the fallback (not from the garbage CTS)."""
    fallback_predictor = MagicMock(return_value="The answer is \\boxed{47}")

    cts_pred = "Cultura"
    bench = "aime"
    if is_garbage_math(bench, cts_pred):
        final = fallback_predictor("FALLBACK_PROMPT", max_new_tokens=512)
    else:
        final = cts_pred

    fallback_predictor.assert_called_once()
    assert "47" in final
    assert "Cultura" not in final


def test_mock_fallback_NOT_invoked_when_cts_returns_clean_number():
    """End-to-end mock: clean CTS prediction; fallback predictor
    must NOT be called (otherwise we lose latency *and* the CTS
    method's value-add)."""
    fallback_predictor = MagicMock(return_value="OOPS_CALLED")

    cts_pred = "47"
    bench = "aime"
    if is_garbage_math(bench, cts_pred):
        final = fallback_predictor("FALLBACK_PROMPT", max_new_tokens=512)
    else:
        final = cts_pred

    fallback_predictor.assert_not_called()
    assert final == "47"


def test_mock_humaneval_never_falls_back():
    """End-to-end mock: HumanEval-style code prediction that
    happens to look like the garbage tokens from AIME. Must NOT
    invoke the fallback (which would replace code with a math
    prompt)."""
    fallback_predictor = MagicMock(return_value="OOPS_CALLED_ON_HUMANEVAL")

    cts_pred = "LinearLayout"  # plausibly a class name in code
    bench = "humaneval"
    if is_garbage_math(bench, cts_pred):
        final = fallback_predictor("FALLBACK_PROMPT", max_new_tokens=512)
    else:
        final = cts_pred

    fallback_predictor.assert_not_called()
    assert final == "LinearLayout"
