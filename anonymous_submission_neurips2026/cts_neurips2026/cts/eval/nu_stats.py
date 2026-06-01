"""ν cross-domain statistics aggregator (paper Table 19).

Paper Table 19 reports per-domain mean ± std of the four meta-policy
outputs ``nu = [nu_expl, nu_tol, nu_temp, nu_act]``, aggregated across
all (step × seed × problem) cells inside each (method, domain) bucket.
The paper highlights two directional claims:

  * ``nu_expl_AIME > nu_expl_GSM8K`` — exploration spikes on AIME because
    high-entropy proof goals demand wider PUCT branching.
  * ``nu_act_GSM8K > nu_act_AIME`` — early-termination fires earlier on
    GSM8K because short numeric arithmetic problems are answered with
    far fewer MCTS iterations.

This module:

  * reads the per-problem JSONL traces emitted by
    ``scripts/run_cts_eval_full.py --nu-trace-dir <dir>``,
  * folds them into a long-form pandas DataFrame,
  * summarises mean ± std per (method, domain) cell,
  * runs Welch's t-tests (Bonferroni-corrected at n=2) for the two
    paper-highlighted directional claims,
  * renders a Markdown table that can be pasted into REVIEWER_FAQ /
    appendix.

It is intentionally pure-Python + pandas + scipy.stats. No torch / no
Gemma weights. The aggregator never crashes on a missing JSONL file or
a JSONL line without a ``nu_trace`` key; instead, it returns an empty
DataFrame and the renderer emits a clear "no data" banner so reviewers
auditing a partial run get an actionable error rather than a stack trace.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Components reported in paper Table 19 (column order is paper-faithful).
NU_COMPONENTS: Tuple[str, ...] = ("nu_expl", "nu_tol", "nu_temp", "nu_act")

# Default benchmark→domain mapping used by `scripts/aggregate_nu_table19.py`.
# Paper Table 19 buckets:
#   math       = {AIME, MATH-500, GSM8K}
#   code       = {HumanEval}
#   reasoning  = {ARC-AGI-Text}
DEFAULT_DOMAIN_MAP: Dict[str, str] = {
    "aime": "math",
    "math500": "math",
    "gsm8k": "math",
    "humaneval": "code",
    "arc_agi_text": "reasoning",
}


# ---------------------------------------------------------------------------
# Long-form aggregation
# ---------------------------------------------------------------------------

def _iter_jsonl_records(path: Path) -> Iterable[dict]:
    """Yield decoded JSON objects from a JSONL file. Tolerates blank lines
    and malformed lines (those are silently skipped)."""
    if not path.exists() or not path.is_file():
        return
    # ``utf-8-sig`` transparently strips the optional UTF-8 BOM that
    # PowerShell's ``Out-File -Encoding utf8`` writes by default; without
    # this, the first line of every PowerShell-emitted JSONL file would
    # have ``\ufeff`` prepended and json.loads would silently drop the
    # whole file.
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def aggregate_nu_traces(
    jsonl_paths: List[Path],
    domain_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Fold per-problem ν JSONL traces into a long-form DataFrame.

    Each input JSONL line is expected to have at least::

        {
          "method": "cts_4nu",
          "benchmark": "aime",
          "seed": 0,
          "problem_id": "aime/0",
          "nu_trace": {
              "nu_expl": [0.93, 1.02, 0.88, ...],   # one float per MCTS step
              "nu_tol":  [...],
              "nu_temp": [...],
              "nu_act":  [...],
          },
        }

    Returns a DataFrame with columns
    ``[method, benchmark, domain, problem_id, seed, nu_component, nu_value]``.
    Each row is ONE per-step ν measurement, so a single problem with
    ``len(nu_trace.nu_expl) == K`` contributes ``4 * K`` rows.

    Lines whose ``benchmark`` is not in ``domain_map`` are dropped (with no
    error) so reviewers can mix runs across paper / non-paper benchmarks.
    """
    if domain_map is None:
        domain_map = DEFAULT_DOMAIN_MAP

    rows: List[dict] = []
    for path in jsonl_paths:
        path = Path(path)
        for rec in _iter_jsonl_records(path):
            nu_trace = rec.get("nu_trace")
            if not isinstance(nu_trace, dict):
                continue
            method = rec.get("method", "")
            benchmark = rec.get("benchmark", "")
            domain = domain_map.get(benchmark)
            if domain is None:
                continue
            problem_id = rec.get("problem_id", "")
            seed = rec.get("seed", 0)
            for comp in NU_COMPONENTS:
                values = nu_trace.get(comp)
                if not isinstance(values, list):
                    continue
                for v in values:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(fv) or math.isinf(fv):
                        continue
                    rows.append({
                        "method": method,
                        "benchmark": benchmark,
                        "domain": domain,
                        "problem_id": problem_id,
                        "seed": int(seed),
                        "nu_component": comp,
                        "nu_value": fv,
                    })

    if not rows:
        return pd.DataFrame(
            columns=[
                "method", "benchmark", "domain", "problem_id",
                "seed", "nu_component", "nu_value",
            ],
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Welch's t-test (one-sided) — pure stdlib so the module stays scipy-optional
# at the call site, but we use scipy when available for higher fidelity.
# ---------------------------------------------------------------------------

def _welch_one_sided_p(a: List[float], b: List[float]) -> float:
    """One-sided Welch's t-test, H1: mean(a) > mean(b).

    Uses ``scipy.stats.ttest_ind(equal_var=False, alternative='greater')`` if
    SciPy is installed; otherwise falls back to a pure-Python implementation
    that uses the Welch-Satterthwaite approximation + the standard normal
    survival function (good to ~1e-3 for df > 30, which is the common case
    for ν aggregation across step × seed × problem).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 1.0
    try:
        from scipy import stats  # type: ignore
        res = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
        p = float(res.pvalue)
        if math.isnan(p):
            return 1.0
        return p
    except Exception:
        pass

    # --- pure-Python Welch fallback ---
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    denom = math.sqrt(var_a / n_a + var_b / n_b)
    if denom <= 0.0:
        return 1.0 if mean_a <= mean_b else 0.0
    t = (mean_a - mean_b) / denom
    # Approximate the survival function of t by N(0,1) for df >> 30.
    # This is the same approximation used by `cts.eval.statistics`.
    p_two = math.erfc(abs(t) / math.sqrt(2.0))
    if t > 0:
        return p_two / 2.0
    return 1.0 - p_two / 2.0


# ---------------------------------------------------------------------------
# Wide summary (rows = method × domain; cols = nu_*_mean±std + p-values)
# ---------------------------------------------------------------------------

def _fmt_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean) or math.isnan(std):
        return ""
    return f"{mean:.3f} ± {std:.3f}"


def summarize_table19(
    df: pd.DataFrame,
    *,
    bonferroni_n: int = 2,
) -> pd.DataFrame:
    """Build the paper Table 19 wide summary.

    Output rows are ``(method, domain)``. Output columns are::

        nu_expl_mean_std, nu_tol_mean_std, nu_temp_mean_std, nu_act_mean_std,
        p_nu_expl_aime_gt_gsm8k,    # raw Welch p
        p_nu_act_gsm8k_gt_aime,     # raw Welch p
        p_nu_expl_aime_gt_gsm8k_corr,  # Bonferroni-corrected at n=`bonferroni_n`
        p_nu_act_gsm8k_gt_aime_corr,
        marker_nu_expl,             # "↑" if AIME > GSM8K significant
        marker_nu_act,              # "↑" if GSM8K > AIME significant

    The two p-values are PER-METHOD (not per cell): they are constant inside
    each method block but copied onto every (method, domain) row so the
    rendered Markdown can place the marker next to the relevant cell without
    a second join.

    The ``↑`` marker rules:
      * placed in the ``math`` row only (the comparison is AIME vs GSM8K);
      * fires iff the Bonferroni-corrected one-sided p is < 0.05.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method", "domain",
                "nu_expl_mean_std", "nu_tol_mean_std",
                "nu_temp_mean_std", "nu_act_mean_std",
                "p_nu_expl_aime_gt_gsm8k", "p_nu_act_gsm8k_gt_aime",
                "p_nu_expl_aime_gt_gsm8k_corr", "p_nu_act_gsm8k_gt_aime_corr",
                "marker_nu_expl", "marker_nu_act",
                "n_steps",
            ],
        )

    methods = sorted(df["method"].unique())
    domains = sorted(df["domain"].unique())

    out_rows: List[dict] = []
    for method in methods:
        m_df = df[df["method"] == method]

        # Per-method directional t-tests across math sub-benchmarks.
        expl_aime = m_df[(m_df["benchmark"] == "aime") &
                         (m_df["nu_component"] == "nu_expl")]["nu_value"].tolist()
        expl_gsm8k = m_df[(m_df["benchmark"] == "gsm8k") &
                          (m_df["nu_component"] == "nu_expl")]["nu_value"].tolist()
        act_aime = m_df[(m_df["benchmark"] == "aime") &
                        (m_df["nu_component"] == "nu_act")]["nu_value"].tolist()
        act_gsm8k = m_df[(m_df["benchmark"] == "gsm8k") &
                         (m_df["nu_component"] == "nu_act")]["nu_value"].tolist()

        p_expl = _welch_one_sided_p(expl_aime, expl_gsm8k)
        p_act = _welch_one_sided_p(act_gsm8k, act_aime)
        p_expl_corr = min(1.0, p_expl * bonferroni_n)
        p_act_corr = min(1.0, p_act * bonferroni_n)

        for domain in domains:
            cell = m_df[m_df["domain"] == domain]
            row: dict = {"method": method, "domain": domain}
            n_steps = 0
            for comp in NU_COMPONENTS:
                vals = cell[cell["nu_component"] == comp]["nu_value"]
                if len(vals) > 0:
                    mean = float(vals.mean())
                    # ddof=1 sample std; degenerate to 0 for n=1.
                    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    row[f"{comp}_mean_std"] = _fmt_mean_std(mean, std)
                    n_steps = max(n_steps, len(vals))
                else:
                    row[f"{comp}_mean_std"] = ""
            row["n_steps"] = int(n_steps)
            row["p_nu_expl_aime_gt_gsm8k"] = float(p_expl)
            row["p_nu_act_gsm8k_gt_aime"] = float(p_act)
            row["p_nu_expl_aime_gt_gsm8k_corr"] = float(p_expl_corr)
            row["p_nu_act_gsm8k_gt_aime_corr"] = float(p_act_corr)
            # Marker rule: only annotate the ``math`` row, since both the
            # AIME and GSM8K populations live inside that bucket.
            if domain == "math" and p_expl_corr < 0.05 and expl_aime and expl_gsm8k:
                row["marker_nu_expl"] = "↑"
            else:
                row["marker_nu_expl"] = ""
            if domain == "math" and p_act_corr < 0.05 and act_aime and act_gsm8k:
                row["marker_nu_act"] = "↑"
            else:
                row["marker_nu_act"] = ""
            out_rows.append(row)

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

_NO_DATA_BANNER = (
    "## Table 19 — ν cross-domain statistics\n\n"
    "_No `nu_trace` data was found in the provided run directories._\n\n"
    "Re-run a CTS evaluation with `--nu-trace-dir <dir>` (or set the\n"
    "`CTS_NU_TRACE_DIR` environment variable) so the dispatcher persists\n"
    "per-problem ν traces, then re-invoke `scripts/aggregate_nu_table19.py`.\n"
)


def render_table19_markdown(summary: pd.DataFrame, out: Path) -> None:
    """Render the wide summary as Markdown into ``out``.

    Always writes a file. When the summary is empty, the file contains the
    "no data" banner so reviewers consuming the path get an actionable
    message rather than a blank document.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if summary is None or summary.empty:
        out.write_text(_NO_DATA_BANNER, encoding="utf-8")
        return

    lines: List[str] = []
    lines.append("## Table 19 — ν cross-domain statistics (mean ± std)")
    lines.append("")
    lines.append(
        "Per-step ν aggregated across (problem × seed × MCTS step) inside each "
        "(method, domain) cell. Markers denote paper-highlighted directional "
        "claims (Welch one-sided, Bonferroni-corrected at n=2):"
    )
    lines.append("")
    lines.append("- `↑ nu_expl` &nbsp;⇒&nbsp; nu_expl(AIME) > nu_expl(GSM8K), "
                 "p_corr < 0.05 (high-entropy AIME drives wider PUCT)")
    lines.append("- `↑ nu_act` &nbsp;⇒&nbsp; nu_act(GSM8K) > nu_act(AIME), "
                 "p_corr < 0.05 (early termination on easy arithmetic)")
    lines.append("")

    header = (
        "| method | domain | nu_expl | nu_tol | nu_temp | nu_act | n_steps |"
    )
    sep = "|---|---|---|---|---|---|---:|"
    lines.append(header)
    lines.append(sep)

    for _, row in summary.iterrows():
        nu_expl_cell = row["nu_expl_mean_std"]
        if row.get("marker_nu_expl"):
            nu_expl_cell = f"{nu_expl_cell} {row['marker_nu_expl']}".strip()
        nu_act_cell = row["nu_act_mean_std"]
        if row.get("marker_nu_act"):
            nu_act_cell = f"{nu_act_cell} {row['marker_nu_act']}".strip()
        lines.append(
            f"| {row['method']} | {row['domain']} | "
            f"{nu_expl_cell} | {row['nu_tol_mean_std']} | "
            f"{row['nu_temp_mean_std']} | {nu_act_cell} | "
            f"{row['n_steps']} |"
        )

    # Per-method one-sided Welch p-value summary (not per-cell, to avoid
    # repeating the same number across every domain row).
    lines.append("")
    lines.append("### Welch one-sided p-values (paper directional claims)")
    lines.append("")
    lines.append("| method | nu_expl(AIME)>nu_expl(GSM8K) raw_p / corr_p | "
                 "nu_act(GSM8K)>nu_act(AIME) raw_p / corr_p |")
    lines.append("|---|---|---|")
    seen: set = set()
    for _, row in summary.iterrows():
        method = row["method"]
        if method in seen:
            continue
        seen.add(method)
        lines.append(
            f"| {method} | "
            f"{row['p_nu_expl_aime_gt_gsm8k']:.4f} / "
            f"{row['p_nu_expl_aime_gt_gsm8k_corr']:.4f} | "
            f"{row['p_nu_act_gsm8k_gt_aime']:.4f} / "
            f"{row['p_nu_act_gsm8k_gt_aime_corr']:.4f} |"
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "NU_COMPONENTS",
    "DEFAULT_DOMAIN_MAP",
    "aggregate_nu_traces",
    "summarize_table19",
    "render_table19_markdown",
]
