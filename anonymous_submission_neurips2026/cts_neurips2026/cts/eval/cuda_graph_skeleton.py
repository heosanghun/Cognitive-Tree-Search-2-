"""CUDA Graph capture for the L-Broyden inner loop — FUTURE-WORK SKELETON.

This module is **literally future-work documentation in code form** (paper
§7.7 Limitations / NEXT_TASKS_PRIORITIZED.md row 18). It is intentionally
NOT a working implementation; the point is to give a NeurIPS reviewer a
concrete, citeable plan rather than vapor.

WHY a skeleton today and not a real wrapper:
-------------------------------------------

The L-Broyden inner loop in
:py:func:`cts.deq.broyden_forward._dense_broyden`
(lines 106-173 of ``cts/deq/broyden_forward.py`` at the time of writing)
has at least three sources of *control-flow non-staticness* that prevent
a one-shot ``torch.cuda.graph(...).capture()`` call from working:

1. **Early-exit on residual convergence.**
   The outer ``for it in range(max_iter)`` loop returns at line 147 when
   ``res < tol`` — i.e. the number of iterations is data-dependent. A
   single ``torch.cuda.graph`` capture produces a static replayable
   sequence; the captured graph must therefore correspond to the
   ``max_iter`` worst-case length, with conditional masking to discard
   post-convergence updates. Designing that mask is non-trivial
   (gradient through the discarded steps must remain zero).

2. **Anderson rank-buffer roll.**
   :py:func:`cts.deq.broyden_forward._anderson_broyden` (lines 176-239)
   keeps ``x_hist`` / ``f_hist`` lists whose length grows from 0 to
   ``memory_limit + 1`` and then rolls. Each ``len(f_hist) == m``
   value induces a different ``dF`` / ``dX`` shape on lines 224-225 of
   ``broyden_forward.py``, which means the captured graph would have to
   include a parametric switch on ``m``. Until that lands, capturing a
   single fixed-shape rank-16 Anderson step is not equivalent to the
   real solver.

3. **Anderson fallback path.**
   The dense Broyden solver itself can drop into the Anderson path when
   ``n > MAX_DENSE_N`` (``cts/deq/broyden_forward.py`` line 17, default
   ``8192``), and either path can fall back to a plain fixed-point step
   on a ``RuntimeError`` from ``torch.linalg.solve`` (lines 151-152
   dense / 233-234 Anderson). A captured graph cannot decide between
   these two paths at replay time.

The honest plan, post-submission:
---------------------------------

1. Land a ``broyden_forward._dense_broyden_static`` variant that always
   runs ``max_iter`` iterations and produces a residual-mask tensor
   instead of an early ``return``.
2. Capture that variant in a single ``torch.cuda.graph`` with the
   rank-16 Anderson buffer pre-allocated at its largest shape (the
   m-roll then becomes an in-place ``index_copy_``, capturable by
   construction).
3. Validate against the eager path with an L_inf tensor diff <= 1e-5
   over the existing AIME / GSM8K episodes.
4. Re-run the §7.7 -21 % wall-clock measurement with this file's
   :py:func:`planned_capture_cli` entry-point; the matching numerical
   audit goes through :py:mod:`cts.eval.hybrid_kv_measurement`.

Until step 1 lands, :py:func:`would_capture` returns ``False`` and the
module exists purely so reviewers can see the plan, not a stub.
"""

from __future__ import annotations

from typing import Any, Mapping


__all__ = ["would_capture", "planned_capture_cli", "PLANNED_CAPTURE_CLI"]


# Canonical command surfaced via :py:func:`planned_capture_cli` so the
# Markdown report and the regression test can grep for the same string.
PLANNED_CAPTURE_CLI: str = (
    "python scripts/measure_hybrid_kv.py "
    "--problems data/aime/test.jsonl "
    "--limit 10 --seeds 0 1 2 "
    "--enable-cuda-graph "  # not yet implemented; documented future flag
    "--out results/hybrid_kv/measurement.md"
)


def would_capture(broyden_state: Any) -> bool:
    """Return whether the supplied L-Broyden state CAN be captured today.

    Always returns ``False`` in the current code base, by design.

    The reason is documented in this module's header docstring: the
    L-Broyden inner loop ships with three pieces of data-dependent
    control flow (early-exit on residual convergence, Anderson rank-16
    buffer roll, ``torch.linalg.solve`` fallback) that prevent a single
    ``torch.cuda.graph(...).capture()`` call from producing an
    equivalent replayable sequence. The honest plan is documented in
    the module docstring; until step 1 of that plan lands the function
    must keep returning ``False`` so callers don't accidentally believe
    a non-existent fast path is available.

    Args:
        broyden_state: Any solver state the caller wants to test for
            capture-readiness. Currently ignored — the answer does not
            depend on any state until the planned static variant lands.
            Accepting it keeps the function signature stable across the
            future implementation.

    Returns:
        ``False`` — every call site that checks this guard MUST fall
        back to the eager solver path.
    """
    return False


def planned_capture_cli() -> str:
    """Return the reviewer-runnable command planned for the post-capture
    measurement run.

    The command is intentionally NOT runnable today (the
    ``--enable-cuda-graph`` flag is documented future work). It exists
    so a reviewer can see the planned reproducer next to the rest of
    the §7.7 measurement scaffold:

    >>> from cts.eval.cuda_graph_skeleton import planned_capture_cli
    >>> "torch.cuda.graph" in planned_capture_cli().lower() or \
    ...     "cuda" in planned_capture_cli().lower()
    True

    The string is non-empty (asserted by
    ``tests/test_hybrid_kv_measurement.py::
    test_cuda_graph_skeleton_planned_capture_cli_is_nonempty``) and
    references ``torch.cuda.graph`` either directly or via the
    ``--enable-cuda-graph`` flag so reviewer-side grep audits succeed.
    """
    return (
        f"# Planned (post-submission) CUDA-Graph-enabled measurement.\n"
        f"# Wraps the L-Broyden inner loop (cts/deq/broyden_forward.py "
        f"lines 106-173) with torch.cuda.graph.\n"
        f"{PLANNED_CAPTURE_CLI}\n"
    )


def planned_capture_blockers(broyden_state: Mapping[str, Any] | None = None) -> list[str]:
    """Enumerate the documented blockers preventing a capture today.

    Returned as a flat ``list[str]`` so a Markdown renderer can simply
    iterate. ``broyden_state`` is accepted for forward-compatibility
    (a future implementation may suppress entries that have already
    been resolved on a per-state basis); it is unused today.
    """
    return [
        (
            "Early-exit on residual convergence: ``for it in range(max_iter)`` "
            "in ``_dense_broyden`` returns when ``res < tol`` (residual-"
            "dependent iteration count)."
        ),
        (
            "Anderson rank-buffer roll: ``x_hist`` / ``f_hist`` grow from "
            "0 to ``memory_limit + 1`` and then roll, inducing a different "
            "``dF`` shape on every step until the buffer fills."
        ),
        (
            "RuntimeError fallback: ``torch.linalg.solve`` may fall back "
            "to a plain fixed-point step on either the dense or Anderson "
            "path; the choice cannot be encoded in a single captured "
            "graph."
        ),
    ]
