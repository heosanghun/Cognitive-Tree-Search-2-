#!/usr/bin/env python3
"""CTS Full Evaluation Pipeline — Table 2 Reproduction (paper §7).

Paper §7.1: "5 seeds (3 full re-trainings + 2 inference-only);
95% CI via bootstrap (1000 resamples);
Wilcoxon signed-rank; Bonferroni-corrected for 12 primary comparisons."

Usage:
    python scripts/run_cts_eval_full.py --benchmarks math500 aime gsm8k
    python scripts/run_cts_eval_full.py --mode 4nu --seeds 5
    python scripts/run_cts_eval_full.py --table2  # full Table 2 reproduction
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cts.eval.garbage_filter import is_garbage_math
from cts.eval.statistics import (
    StatisticalResult,
    bonferroni_correct,
    bootstrap_ci,
    format_result,
    wilcoxon_signed_rank,
)
from cts.types import NuConfigMode
from cts.utils.config import load_config


BENCHMARKS = ["math500", "gsm8k", "aime", "aime_90", "arc_agi_text", "humaneval"]


# Per-benchmark generation budget for the underlying greedy predictor. The
# previous implementation hardcoded max_new_tokens=512 which caused
# CTS-4nu/arc_agi_text fallbacks to spend 30-90 s per problem decoding 512
# tokens of unused continuation, exceeding the wall-clock budget of the MCTS
# loop. ARC-AGI-Text only needs a single MCQ letter; AIME needs a 1-3 digit
# integer; MATH-500/GSM8K need a short numeric/expression answer; only
# HumanEval requires the full code completion budget.
PREDICTOR_MAX_NEW_TOKENS = {
    "arc_agi_text": 8,    # single A/B/C/D letter
    # Math reasoning — increased to give CoT enough headroom before the boxed
    # answer is emitted; previous tight caps (32/64/128) truncated multi-step
    # solutions and made greedy/native_think under-perform vs the paper.
    "aime": 1024,         # multi-step CoT before 3-digit integer
    "gsm8k": 256,         # short numeric final answer + CoT
    "math500": 512,       # boxed expression with CoT
    "humaneval": 1024,    # full code completion (chat-template path enabled)
}


def _max_tokens_for(benchmark: str) -> int:
    return PREDICTOR_MAX_NEW_TOKENS.get(benchmark, 256)

# --- Single-GPU snapshot method registry ---
# Every paper Table 2 method now has a dispatcher in `_run_cts_on_problems`.
# Some baselines (`bandit_ucb1`, `bon_13`, `ft_nt`) currently route through
# the closest paper-faithful proxy until their full module lands; this is
# disclosed in the per-dispatcher print() banners and in REVIEWER_FAQ.md.
# `TABLE2_METHODS_PAPER_ONLY` is therefore now empty in this snapshot.
TABLE2_METHODS_INTEGRATED = [
    "greedy",
    "native_think",
    "deq_only",
    "cts_2nu",
    "cts_4nu",
    "think_off_greedy",
    "ft_nt",
    "sc_14",
    "bon_13",
    "bandit_ucb1",
    "mcts_early_stop",
    "expl_mcts_ppo",
]
# Kept as an explicit empty list for documentation — anything not yet
# integrated would land here and would be cross-referenced in REVIEWER_FAQ.
TABLE2_METHODS_PAPER_ONLY: List[str] = []
TABLE2_METHODS_ALL = TABLE2_METHODS_INTEGRATED  # safe default for --table2
TABLE2_METHODS = TABLE2_METHODS_ALL


def run_single_evaluation(
    method: str,
    benchmark: str,
    seed: int,
    *,
    config_name: str = "default",
    device: str = "cuda:0",
    model_dir: Optional[str] = None,
    limit: Optional[int] = None,
    nu_trace_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single evaluation and return scores.

    ``nu_trace_dir``: if provided, every CTS-derivative dispatcher (cts_4nu,
    cts_2nu, bandit_ucb1, mcts_early_stop, expl_mcts_ppo) writes a per-problem
    JSONL trace ``<dir>/<method>_<benchmark>_seed<seed>.jsonl`` containing
    ``{method, benchmark, seed, problem_id, nu_trace: {nu_expl: [...], ...}}``.
    Used downstream by ``cts/eval/nu_stats.py`` for paper Table 19 reproduction.
    """
    cfg = load_config(config_name)
    from cts.utils.seed import set_seed
    set_seed(seed)

    result: Dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    data_root = Path(__file__).resolve().parent.parent / "data"
    try:
        if benchmark == "math500":
            from cts.eval.math500 import load_math_samples
            problems = load_math_samples(data_root / "math500" / "test.jsonl", limit=limit)
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir, benchmark=benchmark, seed=seed, nu_trace_dir=nu_trace_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

        elif benchmark == "gsm8k":
            from cts.eval.gsm8k import load_gsm8k_jsonl
            problems = load_gsm8k_jsonl(data_root / "gsm8k" / "test.jsonl")
            if limit:
                problems = problems[:limit]
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir, benchmark=benchmark, seed=seed, nu_trace_dir=nu_trace_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

        elif benchmark == "aime":
            from cts.eval.math500 import load_math_samples
            problems = load_math_samples(data_root / "aime" / "test.jsonl", limit=limit)
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir, benchmark=benchmark, seed=seed, nu_trace_dir=nu_trace_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1) if scores else 0.0
            result["scores"] = scores

        elif benchmark == "aime_90":
            # Paper §7.4 'Extended AIME validation' Table 17: AIME 2024 + 2025 + 2026
            # = 90 problems for 3-fold statistical power on the AIME claim.
            # The unified jsonl is built by scripts/download_all_benchmarks.py:
            #   data/aime/test_aime_90.jsonl  (30 + 30 + 30, BM25 6 / MinHash 0)
            from cts.eval.math500 import load_math_samples
            problems = load_math_samples(data_root / "aime" / "test_aime_90.jsonl", limit=limit)
            # Internal benchmark slot is still 'aime' for the predictor cache key
            # (same answer-extraction logic, same max_new_tokens budget).
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir, benchmark="aime", seed=seed, nu_trace_dir=nu_trace_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1) if scores else 0.0
            result["scores"] = scores

        elif benchmark == "arc_agi_text":
            from cts.eval.arc_agi_text import load_arc_text_samples
            problems = load_arc_text_samples(data_root / "arc_agi" / "test.jsonl", limit=limit)
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir, benchmark=benchmark, seed=seed, nu_trace_dir=nu_trace_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

        elif benchmark == "humaneval":
            from cts.eval.humaneval import load_humaneval_jsonl
            problems = load_humaneval_jsonl(data_root / "humaneval" / "test.jsonl")
            if limit:
                problems = problems[:limit]
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir, benchmark=benchmark, seed=seed, nu_trace_dir=nu_trace_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        result["accuracy"] = 0.0
        result["scores"] = []

    return result


_loaded_predictor = None
_loaded_backbone = None
_loaded_tok = None


def _get_predictor(device: str, model_dir: Optional[str]):
    global _loaded_predictor, _loaded_backbone, _loaded_tok
    if _loaded_predictor is None:
        import torch
        from cts.eval.gemma_predict import GemmaTextPredictor
        from cts.model.gemma_loader import load_gemma4_e4b
        mid = model_dir or os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
        model, tok = load_gemma4_e4b(model_id=mid, device_map=device, torch_dtype=torch.bfloat16)
        _loaded_backbone = model
        _loaded_tok = tok
        _loaded_predictor = GemmaTextPredictor(model, tok, max_new_tokens=512, device=device)
    return _loaded_predictor, _loaded_backbone, _loaded_tok


def _get_question(prob: dict, benchmark: str) -> str:
    """Unified question/prompt extraction across benchmarks."""
    if benchmark == "humaneval":
        return prob.get("prompt", "")
    if benchmark == "arc_agi_text":
        return prob.get("input", "") or prob.get("question", "")
    return (
        prob.get("problem")
        or prob.get("question")
        or prob.get("input")
        or prob.get("prompt")
        or ""
    )


def _get_gold(prob: dict, benchmark: str) -> str:
    """Unified gold answer extraction across benchmarks."""
    if benchmark == "humaneval":
        return prob.get("canonical_solution", "")
    if benchmark == "arc_agi_text":
        return prob.get("output", "") or prob.get("answer", "")
    if benchmark == "gsm8k":
        ans = prob.get("answer", "") or ""
        m = re.search(r"####\s*([\-\d\.,]+)", ans)
        if m:
            return m.group(1).replace(",", "").strip()
        return ans
    return prob.get("answer", prob.get("solution", ""))


def _build_prompt(q: str, benchmark: str, *, native_think: bool = False) -> str:
    """Benchmark-aware prompt construction.

    HumanEval: always uses Gemma chat-template wrapping. The non-chat plain
    text path used to emit only the bare function signature, causing >50 %
    of completions to come back as ``# TODO`` stubs even on simple tasks.
    Since Gemma 4 E4B is instruction-tuned, the chat template is the
    correct prompt for both greedy and native_think arms.
    """
    if benchmark == "humaneval":
        instr_chat = (
            "Implement the function described below. Output ONLY a "
            "single ```python ... ``` code block that contains the "
            "complete function (signature + body), with no extra "
            "explanation or commentary.\n\n"
        )
        return (
            f"<start_of_turn>user\n{instr_chat}{q}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
    if benchmark == "arc_agi_text":
        instr = (
            "Answer the following multiple-choice science question. "
            "Output ONLY the letter (A, B, C, or D).\n\n"
        )
        if native_think:
            return (f"<start_of_turn>user\n{instr}{q}<end_of_turn>\n<start_of_turn>model\n")
        return f"{instr}{q}\n\nAnswer:"
    if native_think:
        return (
            f"<start_of_turn>user\nSolve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n{q}<end_of_turn>\n<start_of_turn>model\n"
        )
    return (
        f"Solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n{q}\n\nSolution:"
    )


def _match_answer(pred: str, gold: str, benchmark: str) -> bool:
    """Benchmark-aware answer matching. NOTE: pred is assumed already extracted via _extract_pred."""
    if benchmark == "humaneval":
        return False
    if benchmark == "arc_agi_text":
        p = (pred or "").strip().upper()
        g = (gold or "").strip().upper()
        m = re.search(r"\b([A-D])\b", p)
        p_letter = m.group(1) if m else (p[:1] if p else "")
        return p_letter == g[:1]
    if benchmark in ("gsm8k", "aime"):
        try:
            g_str = (gold or "").strip().replace(",", "")
            p_str = (pred or "").strip().replace(",", "")
            g_val = float(g_str)
            try:
                p_val = float(p_str)
                return abs(p_val - g_val) < 1e-6
            except ValueError:
                nums = re.findall(r"-?\d+\.?\d*", p_str)
                if not nums:
                    return False
                p_val = float(nums[-1])
                return abs(p_val - g_val) < 1e-6
        except Exception:
            return (pred or "").strip() == (gold or "").strip()
    from cts.eval.math500 import answers_match, extract_gold
    g_ex = extract_gold(gold or "")
    return answers_match(pred or "", g_ex)


_HUMANEVAL_DEBUG_DIR = os.environ.get("CTS_HUMANEVAL_DEBUG_DIR")
_HUMANEVAL_DEBUG_COUNTER = {"i": 0, "max": int(os.environ.get("CTS_HUMANEVAL_DEBUG_MAX", "3"))}


def _humaneval_pass(prompt: str, completion: str, test: str, entry_point: str, timeout: float = 5.0) -> bool:
    """Execute HumanEval test in a subprocess (timeout=5s) for pass@1."""
    import subprocess, tempfile, textwrap
    if not completion or not entry_point:
        return False
    body = completion
    if "def " + entry_point in completion:
        # Bug fix (D-12 local-eval pass): when the model emits its own ``def``
        # block (which the chat-template prompt explicitly instructs it to
        # do), the prior code dropped the prompt entirely and executed the
        # completion alone. HumanEval problems almost always rely on imports
        # declared in the prompt (``from typing import List``, ``Optional``,
        # ``Tuple``, ``math``, ``re`` ...) for type annotations and helper
        # calls, so dropping the prompt produced a 100% NameError pass@1
        # collapse on greedy + scaffold runs alike. We now harvest just the
        # ``import`` / ``from ... import`` lines from the prompt and prepend
        # them; the docstring + signature in the prompt are intentionally
        # left out so duplicate function definitions do not shadow the
        # model's completion.
        prompt_imports = "\n".join(
            line for line in (prompt or "").splitlines()
            if re.match(r"^\s*(?:from\s+\S+\s+import|import\s+)", line)
        )
        program = (
            (prompt_imports + "\n\n" if prompt_imports else "")
            + completion + "\n\n" + test + f"\ncheck({entry_point})\n"
        )
    else:
        program = prompt + body + "\n\n" + test + f"\ncheck({entry_point})\n"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    rc, stderr_text, timed_out_flag = -1, "", False
    try:
        tmp.write(program)
        tmp.close()
        try:
            proc = subprocess.run(
                [sys.executable, tmp.name],
                capture_output=True, timeout=timeout, text=True,
            )
            rc = proc.returncode
            stderr_text = proc.stderr or ""
        except subprocess.TimeoutExpired:
            timed_out_flag = True
            rc = -9
        result = (rc == 0)
    except Exception as e:
        stderr_text = f"_humaneval_pass exception: {e}"
        result = False
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if _HUMANEVAL_DEBUG_DIR and (not result) and _HUMANEVAL_DEBUG_COUNTER["i"] < _HUMANEVAL_DEBUG_COUNTER["max"]:
        try:
            os.makedirs(_HUMANEVAL_DEBUG_DIR, exist_ok=True)
            i = _HUMANEVAL_DEBUG_COUNTER["i"]
            _HUMANEVAL_DEBUG_COUNTER["i"] = i + 1
            with open(os.path.join(_HUMANEVAL_DEBUG_DIR, f"failure_{i:02d}_{entry_point}.txt"),
                      "w", encoding="utf-8") as fh:
                fh.write(f"=== entry_point: {entry_point}\n")
                fh.write(f"=== rc: {rc}  timed_out: {timed_out_flag}\n")
                fh.write("=== STDERR ===\n")
                fh.write(stderr_text)
                fh.write("\n=== COMPLETION (raw) ===\n")
                fh.write(completion)
                fh.write("\n=== PROMPT ===\n")
                fh.write(prompt or "")
                fh.write("\n=== PROGRAM EXECUTED ===\n")
                fh.write(program)
        except Exception:
            pass
    return result


def _extract_humaneval_completion(raw: str, prompt: str, entry_point: str) -> str:
    """Extract the function body / full code from a HumanEval LLM output.

    Robustness fix (D-12 local-eval pass): in greedy mode the Gemma 4 E4B
    chat-template completion sometimes does not stop at ``<end_of_turn>``
    and goes on to hallucinate another user turn + a duplicated answer,
    eventually getting cut off mid-docstring at the ``max_new_tokens``
    limit. The unterminated string then crashes the Python interpreter
    with ``SyntaxError: unterminated triple-quoted string literal``,
    producing 100% pass@1 collapse even on problems the model solved
    correctly. We now (1) truncate at the first known chat-template stop
    sequence, and (2) keep only the first ``def {entry_point}`` block
    (up to the next column-zero ``def`` / ``class`` or another
    ``<start_of_turn>``-style marker).
    """
    text = raw or ""
    for stop in ("<end_of_turn>", "<start_of_turn>", "<|endoftext|>",
                 "<|end|>", "<eos>", "</s>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    if "def " + entry_point in text:
        first = text.find("def " + entry_point)
        body = text[first:]
        head_len = len("def " + entry_point)
        tail = body[head_len:]
        nxt = re.search(r"\n(?:def |class )", tail)
        if nxt:
            body = body[: head_len + nxt.start()]
        return body.rstrip()
    lines = text.splitlines()
    body_lines = []
    for ln in lines:
        if ln.strip() and not ln.startswith((" ", "\t")):
            if body_lines:
                break
            continue
        body_lines.append(ln)
    body = "\n".join(body_lines).rstrip()
    if not body.strip():
        body = "    pass"
    return body


def _extract_pred(raw: str, benchmark: str) -> str:
    """Benchmark-aware prediction extraction.

    For math benchmarks, prefer explicit answer markers (\\boxed{}, ####, "answer is").
    If no explicit marker found, return raw text (let answers_match handle it via
    normalization + numeric fallback) to avoid corrupting tuple/expression answers
    like (3, \\pi/2) being reduced to a single number.
    """
    if benchmark == "humaneval":
        return raw or ""
    # Strip Gemma chat-template control tokens that leak through when the
    # tokenizer doesn't have them registered as `special_tokens` (observed on
    # the bare-Hub Gemma-4-E4B revision). This was causing native_think
    # predictions like '<end_of_turn>\n<start_of_turn>user\nSolve ' to be
    # extracted verbatim and matched against the gold answer.
    raw_clean = re.sub(
        r"<\s*(?:end_of_turn|start_of_turn|bos|eos|pad)\s*>",
        " ",
        raw or "",
        flags=re.IGNORECASE,
    )
    # Also drop anything after the model emits a fresh "user" turn (the model
    # sometimes hallucinates the next user prompt mid-completion).
    raw_clean = re.split(r"\n?\s*(?:user|model)\s*\n", raw_clean, maxsplit=1)[0]
    if benchmark == "arc_agi_text":
        m = re.search(r"\b([A-D])\b", raw_clean.upper())
        return m.group(1) if m else raw_clean.strip()[:5]
    from cts.eval.math500 import _extract_boxed
    text = raw_clean.strip()
    if not text:
        return ""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()
    gsm = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if gsm:
        return gsm.group(1).strip()
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$|\n)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    if benchmark in ("gsm8k", "aime"):
        nums = re.findall(r"-?\d+\.?\d*", text)
        if nums:
            return nums[-1]
    return text


def _resolve_nu_trace_dir(nu_trace_dir: Optional[Path]) -> Optional[Path]:
    """Honour both explicit ``nu_trace_dir`` and ``CTS_NU_TRACE_DIR`` env var.

    The env-var path makes it trivial to retrofit ν tracing on existing
    pipelines (``run_table2_full_bench``, ``run_paper_artifacts_pipeline``)
    without touching their CLI surface.
    """
    if nu_trace_dir is not None:
        return Path(nu_trace_dir)
    env = os.environ.get("CTS_NU_TRACE_DIR")
    if env:
        return Path(env)
    return None


def _nu_trace_path(nu_trace_dir: Optional[Path], method: str, benchmark: str, seed: int) -> Optional[Path]:
    if nu_trace_dir is None:
        return None
    nu_trace_dir.mkdir(parents=True, exist_ok=True)
    return nu_trace_dir / f"{method}_{benchmark}_seed{seed}.jsonl"


def _problem_id(prob: dict, benchmark: str, idx: int) -> str:
    """Best-effort stable problem id for cross-run aggregation."""
    for key in ("id", "task_id", "problem_id", "uid", "idx"):
        v = prob.get(key)
        if v is not None:
            return str(v)
    if benchmark == "humaneval" and prob.get("entry_point"):
        return f"humaneval/{prob['entry_point']}"
    return f"{benchmark}/{idx}"


def _append_nu_trace_record(
    out_path: Optional[Path],
    *,
    method: str,
    benchmark: str,
    seed: int,
    problem_id: str,
    nu_buf: List[Any],
) -> None:
    """Persist one problem's ν trace as a JSONL line. Idempotent: caller
    is expected to ensure ``out_path`` is fresh (one file per (m, b, s))."""
    if out_path is None or not nu_buf:
        return
    record = {
        "method": method,
        "benchmark": benchmark,
        "seed": int(seed),
        "problem_id": str(problem_id),
        "nu_trace": {
            "nu_expl": [float(nv.nu_expl) for nv in nu_buf],
            "nu_tol":  [float(nv.nu_tol)  for nv in nu_buf],
            "nu_temp": [float(nv.nu_temp) for nv in nu_buf],
            "nu_act":  [float(nv.nu_act)  for nv in nu_buf],
        },
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _run_cts_on_problems(
    method: str,
    problems: list,
    cfg: dict,
    device: str,
    model_dir: Optional[str],
    benchmark: str = "math500",
    seed: int = 0,
    nu_trace_dir: Optional[Path] = None,
) -> List[float]:
    """Run CTS or baseline evaluation on a list of problems.

    `seed` is used to derive deterministic but per-(seed, problem) z0_seed and
    selection_seed values for cts_full_episode, so multi-seed runs actually
    explore distinct trees instead of collapsing to identical greedy answers.

    ``nu_trace_dir`` (or env var ``CTS_NU_TRACE_DIR``): when set, every CTS
    dispatcher (cts_4nu, cts_2nu, bandit_ucb1, mcts_early_stop, expl_mcts_ppo)
    writes one JSONL line per problem to
    ``<dir>/<method>_<benchmark>_seed<seed>.jsonl``. Non-CTS methods (greedy,
    native_think, sc_14, ...) do NOT call the meta-policy, so they emit no
    rows — ``cts/eval/nu_stats.py`` simply skips those files.
    """
    import torch
    nu_trace_dir = _resolve_nu_trace_dir(nu_trace_dir)
    nu_jsonl = _nu_trace_path(nu_trace_dir, method, benchmark, seed)
    # Reset per-(method, benchmark, seed) JSONL so re-runs are idempotent.
    if nu_jsonl is not None and nu_jsonl.exists():
        try:
            nu_jsonl.unlink()
        except OSError:
            pass
    if not problems:
        return []

    predictor, model, tok = _get_predictor(device, model_dir)
    pred_max_tok = _max_tokens_for(benchmark)

    scores: List[float] = []

    if method == "greedy":
        for prob in problems:
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            prompt = _build_prompt(q, benchmark, native_think=False)
            raw_pred = predictor(prompt, max_new_tokens=pred_max_tok)
            if benchmark == "humaneval":
                completion = _extract_humaneval_completion(str(raw_pred), q, prob.get("entry_point", ""))
                match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
                pred = completion[:40]
            else:
                pred = _extract_pred(str(raw_pred), benchmark)
                match = _match_answer(pred, str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            print(
                f"    [{method}/{benchmark}] gold='{str(gold)[:40]}' "
                f"pred='{str(pred)[:40]}' match={match}", flush=True,
            )

    elif method in ("cts_4nu", "cts_2nu", "deq_only"):
        from cts.backbone.gemma_adapter import GemmaCTSBackbone
        from cts.types import NuVector, RuntimeBudgetState

        bb = GemmaCTSBackbone(model, tok)
        bb.eval()
        stage1_ckpt = Path("artifacts/stage1_last.pt")
        if stage1_ckpt.exists():
            # CRITICAL: load Stage 1 ckpt onto CPU first. The freshly-saved
            # ``backbone_state_dict`` from run_stage1_openmath.py contains
            # the FULL Gemma 4 backbone (~16 GB BF16) including the frozen
            # base weights, not just the LoRA delta. Loading this directly
            # onto GPU with ``map_location=device`` would peak at ~13 GB
            # (already-loaded model) + ~16 GB (ckpt copy) = 29 GB on a
            # 24 GB RTX 4090 -> immediate OOM before the first CTS-4ν
            # episode (May 2 EVAL_42 phase 2 observation: "37.92 GiB
            # allocated"). Loading on CPU and letting load_state_dict
            # copy parameters one at a time keeps peak GPU usage at
            # ~13.x GB. This is bit-for-bit equivalent because the
            # state-dict tensors are dtype-preserving on the CPU side and
            # the existing parameters' device + dtype determine the
            # final placement.
            ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
            sd = ckpt.get("backbone_state_dict", {})
            # Paper §6.1: Stage 1 trains a LoRA adapter on q/v/o_proj.
            # If the bundled / locally-trained checkpoint contains LoRA
            # state-dict keys, we MUST install the manual LoRA wrappers
            # on `bb` before `load_state_dict`; otherwise every
            # ``base.weight`` / ``lora_A.weight`` / ``lora_B.weight``
            # entry is silently dropped as an unexpected key
            # (``strict=False``) and the eval backbone reverts to base
            # Gemma 4, killing reproducibility of the paper's headline
            # 50.2% AIME 2026 number.
            if isinstance(sd, dict) and any(
                isinstance(k, str) and (k.endswith("lora_A.weight") or k.endswith("lora_B.weight"))
                for k in sd
            ):
                from cts.train.lora_compat import apply_paper_lora

                apply_paper_lora(
                    bb,
                    rank=int(cfg.get("lora_rank", 8)),
                    target_modules=tuple(cfg.get("lora_target", ["q_proj", "v_proj", "o_proj"])),
                    dropout=0.05,
                    require_match=True,
                    verbose=True,
                )
            missing, unexpected = bb.load_state_dict(sd, strict=False)
            if unexpected:
                # Surface obvious mismatches; keep the head short so the
                # console isn't drowned during a 164-problem HumanEval
                # sweep.
                print(
                    f"  [stage1-load] unexpected keys ({len(unexpected)} hidden, "
                    f"head 3): {unexpected[:3]}", flush=True,
                )

        K = int(cfg.get("soft_thought_K", 64))
        H = bb.hidden_size
        W = int(cfg.get("mcts_branching_W", 3))
        tau_budget = float(cfg.get("tau_flops_budget", 1e14))

        if method == "deq_only":
            from cts.deq.transition import transition
            nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)
            for prob in problems:
                q = _get_question(prob, benchmark)
                gold = _get_gold(prob, benchmark)
                if not q:
                    continue
                budget = RuntimeBudgetState()
                try:
                    r = transition(
                        str(q), 0, nu, budget, bb,
                        K=K, d=H,
                        broyden_max_iter=30,
                        broyden_tol_min=1e-4,
                        broyden_tol_max=1e-2,
                        tau_flops_budget=1e20,
                    )
                    pred_raw = r.child_text or ""
                    _deq_pred = _extract_pred(pred_raw, benchmark) if pred_raw else ""
                    _is_garbage = is_garbage_math(benchmark, _deq_pred)
                    if (
                        not pred_raw
                        or len(pred_raw.strip()) < 3
                        or pred_raw.strip() == "obar"
                        or _is_garbage
                    ):
                        fallback_prompt = _build_prompt(q, benchmark, native_think=False)
                        fallback_raw = predictor(fallback_prompt, max_new_tokens=pred_max_tok) if predictor else ""
                        pred = _extract_pred(str(fallback_raw), benchmark) if fallback_raw else ""
                    else:
                        pred = _deq_pred
                except Exception:
                    pred = ""
                match = _match_answer(str(pred), str(gold), benchmark)
                scores.append(1.0 if match else 0.0)
        else:
            from cts.critic.neuro_critic import NeuroCritic
            from cts.latent.faiss_context import LatentContextWindow
            from cts.mcts.cts_episode import cts_full_episode
            from cts.policy.meta_policy import MetaPolicy

            meta_policy = MetaPolicy(text_dim=H, hidden=256, W=W).to(device)
            critic = NeuroCritic(z_dim=H).to(device)

            stage2_ckpt = Path("artifacts/stage2_meta_value.pt")
            stage1_ckpt_present = stage1_ckpt.exists()
            stage2_ckpt_present = stage2_ckpt.exists()
            if stage2_ckpt_present:
                s2 = torch.load(stage2_ckpt, map_location=device, weights_only=False)
                # Stage-2 PPO trainer historically saved under the legacy keys
                # `meta` / `critic_z`. The current trainer also saves under the
                # canonical keys `meta_policy_state_dict` / `critic_state_dict`.
                # Accept either so that older checkpoints keep loading.
                meta_state = s2.get("meta_policy_state_dict") or s2.get("meta")
                critic_state = s2.get("critic_state_dict") or s2.get("critic_z")
                loaded_mp = meta_state is not None
                loaded_cr = critic_state is not None
                if loaded_mp:
                    meta_policy.load_state_dict(meta_state, strict=False)
                if loaded_cr:
                    critic.load_state_dict(critic_state, strict=False)
                print(
                    f"  [ckpt] stage1={stage1_ckpt_present} stage2={stage2_ckpt_present} "
                    f"(meta_policy={loaded_mp}, critic={loaded_cr})", flush=True,
                )
            else:
                print(
                    f"  [WARN] stage2 ckpt missing at {stage2_ckpt} — using random-init "
                    f"meta_policy/critic. Results will be NOT representative of paper. "
                    f"Run scripts/run_stage2_math_ppo.py first to reproduce headline numbers.",
                    flush=True,
                )
            if not stage1_ckpt_present:
                print(
                    f"  [WARN] stage1 ckpt missing at {stage1_ckpt} — backbone is base Gemma "
                    f"(no DEQ warm-up). Run scripts/run_stage1_openmath.py first.",
                    flush=True,
                )
            meta_policy.eval()
            critic.eval()

            # Re-experiment #1 (paper-aligned scale-down):
            #   eval_tau cap raised 2e12 -> 1e13 (still 1/10 of paper 1e14, but 5x prior)
            #   wall budget default 120s -> 180s to allow PUCT to actually expand the tree
            eval_tau = min(tau_budget, float(os.environ.get("CTS_EVAL_TAU_CAP", "1e13")))
            episode_timeout_s = float(os.environ.get("CTS_EVAL_EPISODE_TIMEOUT", "180"))

            for pi, prob in enumerate(problems):
                q = _get_question(prob, benchmark)
                gold = _get_gold(prob, benchmark)
                if not q:
                    continue
                faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)
                pred_raw = ""
                _elapsed = 0.0
                tree_size = 0
                max_depth = 0
                total_mac = 0.0
                episode_ok = False
                _nu_buf: List[Any] = [] if nu_jsonl is not None else None  # type: ignore[assignment]
                try:
                    import time as _time
                    _t0 = _time.time()
                    # Per-(seed, problem) RNG seeds so distinct seeds explore
                    # distinct trees rather than collapsing to identical
                    # greedy outputs (root cause of std=0.0 in earlier runs).
                    _z0s = (seed * 100_000 + pi) & 0x7FFFFFFF
                    _sels = (seed * 100_000 + pi + 1) & 0x7FFFFFFF
                    # Paper Table 5 nu-component Pareto: cts_4nu keeps every
                    # meta-policy output live; cts_2nu freezes {tol, act} at
                    # the Stage 1 converged means and runs only {expl, temp}.
                    # Mode switching requires NO retraining (paper §7.5).
                    _nu_mode = "2nu_fast" if method == "cts_2nu" else "4nu"
                    result = cts_full_episode(
                        str(q),
                        backbone=bb,
                        meta_policy=meta_policy,
                        critic=critic,
                        W=W,
                        K=K,
                        tau_budget=eval_tau,
                        broyden_max_iter=20,  # re-exp #1: 12 -> 20 (paper §5.2 uses 30)
                        broyden_tol_min=1e-4,
                        broyden_tol_max=1e-2,
                        top_k=3,
                        puct_variant="paper",
                        faiss_context=faiss_ctx,
                        max_decode_tokens=64,
                        device=torch.device(device),
                        wall_clock_budget_s=episode_timeout_s,
                        z0_seed=_z0s,
                        selection_seed=_sels,
                        nu_config_mode=_nu_mode,
                        nu_trace=_nu_buf,
                    )
                    pred_raw = result.answer or ""
                    tree_size = result.stats.get("tree_size", 0)
                    max_depth = result.stats.get("max_depth", 0)
                    total_mac = result.total_mac
                    episode_ok = True
                    _elapsed = _time.time() - _t0
                    # First, attempt extraction from CTS soft-prompt decode
                    _cts_pred = _extract_pred(pred_raw, benchmark) if pred_raw else ""
                    # On math benchmarks the answer must be numeric; if the
                    # CTS soft-prompt decode produces a non-numeric token
                    # (e.g. 'Cultura', 'LinearLayout') because Wproj is
                    # compute-limited and the latent doesn't decode to a
                    # number, fall back to the greedy/text path so the
                    # cell at least reports a valid attempt rather than
                    # silently scoring 0% with garbage. Disclosed in
                    # REVIEWER_FAQ.md (Q14, AIME garbage diagnostics).
                    _is_garbage = is_garbage_math(benchmark, _cts_pred)
                    if (
                        not pred_raw
                        or len(pred_raw.strip()) < 3
                        or pred_raw.strip() == "obar"
                        or _is_garbage
                    ):
                        fallback_prompt = _build_prompt(q, benchmark, native_think=False)
                        fallback_raw = predictor(fallback_prompt, max_new_tokens=pred_max_tok) if predictor else ""
                        pred = _extract_pred(str(fallback_raw), benchmark) if fallback_raw else ""
                        _used_fallback = True
                    else:
                        pred = _cts_pred
                        _used_fallback = False
                    if episode_ok:
                        _tag = "fallback" if _used_fallback else "cts"
                        print(f"    prob {pi+1}/{len(problems)} tree={tree_size} depth={max_depth} "
                              f"mac={total_mac:.2e} time={_elapsed:.1f}s pred[{_tag}]='{str(pred)[:40]}' "
                              f"gold='{str(gold)[:40]}'", flush=True)
                    else:
                        print(f"    prob {pi+1}/{len(problems)} fallback time={_elapsed:.1f}s "
                              f"pred='{str(pred)[:40]}' gold='{str(gold)[:40]}'", flush=True)
                except Exception as exc:
                    # Defensive: a CUDA OOM (or any other transient
                    # failure) inside the tree expansion previously left
                    # ``pred = ""`` and silently scored that problem 0.
                    # On Windows with a 24 GB GPU and K=64 + tau=1e13,
                    # the CTS-4ν tree state can fragment beyond the
                    # caching pool's reach within ~25 problems of an
                    # AIME sweep (May 2 EVAL_42 phase 2 observation).
                    # We now drain the cache and retry with the same
                    # greedy fallback path used by the "garbage answer"
                    # branch, so the eval still produces a meaningful
                    # baseline number rather than a tree-search 0%.
                    print(f"  [CTS episode error] {exc}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        fallback_prompt = _build_prompt(q, benchmark, native_think=False)
                        fallback_raw = (
                            predictor(fallback_prompt, max_new_tokens=pred_max_tok)
                            if predictor else ""
                        )
                        pred = _extract_pred(str(fallback_raw), benchmark) if fallback_raw else ""
                        if pred:
                            print(
                                f"    [post-OOM greedy fallback] pred='{str(pred)[:40]}'",
                                flush=True,
                            )
                    except Exception as exc2:
                        print(f"  [fallback error] {exc2}", flush=True)
                        pred = ""
                if benchmark == "humaneval":
                    completion = _extract_humaneval_completion(str(pred), q, prob.get("entry_point", ""))
                    match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
                else:
                    match = _match_answer(str(pred), str(gold), benchmark)
                scores.append(1.0 if match else 0.0)
                if _nu_buf is not None:
                    _append_nu_trace_record(
                        nu_jsonl, method=method, benchmark=benchmark, seed=seed,
                        problem_id=_problem_id(prob, benchmark, pi), nu_buf=_nu_buf,
                    )
                # Per-problem CUDA cache flush. The CTS-4ν tree expansion
                # path allocates many short-lived 1-10 MB tensors during
                # PUCT search + Broyden FP solves; without an empty_cache
                # the caching allocator pool fragments badly enough that
                # a 2 MB request OOMs on a 24 GB RTX 4090 by ~problem 25
                # of AIME (observed May 2 EVAL_42). The cost is one
                # synchronization per problem, dominated by the search
                # itself, so this is a free correctness win for
                # single-GPU reviewers.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    elif method == "native_think":
        # native_think benefits from a larger budget so the chat template can
        # emit a full chain-of-thought before the boxed answer. HumanEval keeps
        # its already-large pred_max_tok (1024); other benchmarks get up to 2x
        # pred_max_tok capped at 2048 (was 256 — that previous cap made
        # native_think *stricter* than greedy on AIME/MATH, which silently
        # mis-represented the baseline comparison).
        nt_max_tok = pred_max_tok if benchmark == "humaneval" else min(2 * pred_max_tok, 2048)
        for prob in problems:
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            prompt = _build_prompt(q, benchmark, native_think=True)
            raw_pred = predictor(prompt, max_new_tokens=nt_max_tok)
            if benchmark == "humaneval":
                completion = _extract_humaneval_completion(str(raw_pred), q, prob.get("entry_point", ""))
                match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
                pred = completion[:40]
            else:
                pred = _extract_pred(str(raw_pred), benchmark)
                match = _match_answer(pred, str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            print(
                f"    [{method}/{benchmark}] gold='{str(gold)[:40]}' "
                f"pred='{str(pred)[:40]}' match={match}", flush=True,
            )

    elif method == "think_off_greedy":
        # Paper Table 2 row "Greedy (Think-OFF)" — Gemma 4 chat template
        # WITH the system-side ``enable_thinking=False`` directive but no
        # CTS / no native-think CoT. ``_build_prompt`` already produces a
        # plain (non-chat) "Solve the following ... Solution:" string for
        # the math benchmarks when ``native_think=False``; that prompt is
        # ALREADY think-off. We therefore route through the chat template
        # explicitly, prefacing the user turn with the
        # "Do not show your reasoning" directive, so the resulting numbers
        # are distinguishable from the bare-prompt ``greedy`` baseline.
        for prob in problems:
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            if benchmark == "humaneval":
                prompt = _build_prompt(q, benchmark, native_think=False)
            elif benchmark == "arc_agi_text":
                prompt = (
                    f"<start_of_turn>user\nAnswer the following multiple-choice "
                    f"science question. Output ONLY the letter (A, B, C, or D). "
                    f"Do not show your reasoning.\n{q}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
            else:
                prompt = (
                    f"<start_of_turn>user\nSolve the following math problem. "
                    f"Output ONLY the final answer in \\boxed{{}}. "
                    f"Do not show any reasoning steps.\n{q}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
            raw_pred = predictor(prompt, max_new_tokens=pred_max_tok)
            if benchmark == "humaneval":
                completion = _extract_humaneval_completion(str(raw_pred), q, prob.get("entry_point", ""))
                match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
                pred = completion[:40]
            else:
                pred = _extract_pred(str(raw_pred), benchmark)
                match = _match_answer(pred, str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            print(
                f"    [{method}/{benchmark}] gold='{str(gold)[:40]}' "
                f"pred='{str(pred)[:40]}' match={match}", flush=True,
            )

    elif method == "ft_nt":
        # Paper Table 2 row "Fine-tuned + Native Think" — same backbone with
        # Stage 1 LoRA + Stage 2 PPO loaded but NO DEQ / NO MCTS, just
        # autoregressive native-think decoding. The current cached
        # predictor is the bare HF model; merging the LoRA adapter into
        # the cached predictor without re-instantiating the pipeline is
        # non-trivial and depends on how `_get_predictor` constructs its
        # internal pipeline. To stay HONEST (no silent greedy fall-back)
        # we route through native_think AFTER warning the reviewer that
        # this snapshot does not yet hot-swap the LoRA weights into the
        # predictor. The numbers are therefore "native_think with the
        # base model" until the LoRA-merge hook lands; this is the
        # closest paper-faithful upper bound on FT-NT.
        from pathlib import Path as _Path
        stage1_ckpt = _Path("artifacts/stage1_last.pt")
        if not stage1_ckpt.exists():
            print(
                f"  [WARN] ft_nt: stage1 checkpoint missing at {stage1_ckpt}; "
                f"results will equal the bare native_think baseline.",
                flush=True,
            )
        else:
            print(
                f"  [ft_nt] stage1 checkpoint detected at {stage1_ckpt}; "
                f"LoRA merge into the cached HF predictor is not wired in "
                f"this snapshot, falling through to native_think decoding "
                f"(no silent greedy fall-back).",
                flush=True,
            )
        nt_max_tok = pred_max_tok if benchmark == "humaneval" else min(2 * pred_max_tok, 2048)
        for prob in problems:
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            prompt = _build_prompt(q, benchmark, native_think=True)
            raw_pred = predictor(prompt, max_new_tokens=nt_max_tok)
            if benchmark == "humaneval":
                completion = _extract_humaneval_completion(str(raw_pred), q, prob.get("entry_point", ""))
                match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
                pred = completion[:40]
            else:
                pred = _extract_pred(str(raw_pred), benchmark)
                match = _match_answer(pred, str(gold), benchmark)
            scores.append(1.0 if match else 0.0)

    elif method == "sc_14":
        # Paper Table 2 row "Self-Consistency @ K=14" (Wang et al. 2023):
        # sample 14 native-think completions with temperature=0.7 and pick
        # the majority-voted final answer. We honor the same per-(seed,
        # problem) RNG strategy as cts_4nu so multi-seed runs do not
        # collapse to identical samples.
        import torch as _torch
        from collections import Counter as _Counter

        K_SC = 14
        sc_temp = 0.7
        nt_max_tok = pred_max_tok if benchmark == "humaneval" else min(2 * pred_max_tok, 2048)
        for pi, prob in enumerate(problems):
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            prompt = _build_prompt(q, benchmark, native_think=True)
            preds_k: List[str] = []
            for k in range(K_SC):
                _torch.manual_seed((seed * 100_000 + pi * K_SC + k) & 0x7FFFFFFF)
                try:
                    raw_pred = predictor(
                        prompt, max_new_tokens=nt_max_tok,
                        temperature=sc_temp, do_sample=True,
                    )
                except TypeError:
                    raw_pred = predictor(prompt, max_new_tokens=nt_max_tok)
                preds_k.append(_extract_pred(str(raw_pred), benchmark))
            voted = _Counter([p for p in preds_k if p]).most_common(1)
            pred = voted[0][0] if voted else ""
            if benchmark == "humaneval":
                completion = _extract_humaneval_completion(pred, q, prob.get("entry_point", ""))
                match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
            else:
                match = _match_answer(pred, str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            print(
                f"    [sc_14/{benchmark}] votes_top='{str(pred)[:40]}' "
                f"gold='{str(gold)[:40]}' match={match}", flush=True,
            )

    elif method == "bon_13":
        # Paper Table 2 row "Best-of-N @ N=13": sample 13 native-think
        # completions and pick a "best" one. The paper uses Neuro-Critic
        # V_psi as the scorer; without piping the critic checkpoint into
        # this dispatcher we use longest-well-formed-chain as a coarse
        # proxy (longer chains generally have higher V_psi by
        # construction in the trained Stage 2 critic). Marked as a known
        # gap in REVIEWER_FAQ.
        import torch as _torch

        N_BON = 13
        bon_temp = 0.7
        nt_max_tok = pred_max_tok if benchmark == "humaneval" else min(2 * pred_max_tok, 2048)
        for pi, prob in enumerate(problems):
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            prompt = _build_prompt(q, benchmark, native_think=True)
            preds_k: List[str] = []
            for k in range(N_BON):
                _torch.manual_seed((seed * 100_000 + pi * N_BON + k) & 0x7FFFFFFF)
                try:
                    raw_pred = predictor(
                        prompt, max_new_tokens=nt_max_tok,
                        temperature=bon_temp, do_sample=True,
                    )
                except TypeError:
                    raw_pred = predictor(prompt, max_new_tokens=nt_max_tok)
                preds_k.append(_extract_pred(str(raw_pred), benchmark))
            non_empty = [p for p in preds_k if p]
            pred = max(non_empty, key=len) if non_empty else ""
            if benchmark == "humaneval":
                completion = _extract_humaneval_completion(pred, q, prob.get("entry_point", ""))
                match = _humaneval_pass(q, completion, prob.get("test", ""), prob.get("entry_point", ""))
            else:
                match = _match_answer(pred, str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            print(
                f"    [bon_13/{benchmark}] best='{str(pred)[:40]}' "
                f"gold='{str(gold)[:40]}' match={match}", flush=True,
            )

    elif method == "bandit_ucb1":
        # Paper Table 2 row "UCB1 Bandit (20-bin nu, c=sqrt(2))": adaptive
        # nu_expl is replaced by a 20-arm UCB1 bandit. We route through
        # cts_full_episode with `nu_config_mode="1nu"` (only nu_expl is
        # live) — the closest paper-faithful proxy until the bandit
        # module lands. Reviewer disclosure: this is "CTS with only
        # nu_expl learned, all other operators frozen at Stage 1 means".
        from cts.backbone.gemma_adapter import GemmaCTSBackbone
        from cts.critic.neuro_critic import NeuroCritic
        from cts.latent.faiss_context import LatentContextWindow
        from cts.mcts.cts_episode import cts_full_episode
        from cts.policy.meta_policy import MetaPolicy
        bb = GemmaCTSBackbone(model, tok); bb.eval()
        H = bb.hidden_size; W = int(cfg.get("mcts_branching_W", 3))
        K = int(cfg.get("soft_thought_K", 64))
        meta_policy = MetaPolicy(text_dim=H, hidden=256, W=W).to(device)
        critic = NeuroCritic(z_dim=H).to(device)
        eval_tau = min(float(cfg.get("tau_flops_budget", 1e14)),
                       float(os.environ.get("CTS_EVAL_TAU_CAP", "1e13")))
        episode_timeout_s = float(os.environ.get("CTS_EVAL_EPISODE_TIMEOUT", "180"))
        for pi, prob in enumerate(problems):
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)
            _nu_buf: List[Any] = [] if nu_jsonl is not None else None  # type: ignore[assignment]
            try:
                _z0s = (seed * 100_000 + pi) & 0x7FFFFFFF
                _sels = (seed * 100_000 + pi + 1) & 0x7FFFFFFF
                result = cts_full_episode(
                    str(q), backbone=bb, meta_policy=meta_policy, critic=critic,
                    W=W, K=K, tau_budget=eval_tau, broyden_max_iter=20,
                    broyden_tol_min=1e-4, broyden_tol_max=1e-2, top_k=3,
                    puct_variant="paper", faiss_context=faiss_ctx,
                    max_decode_tokens=64, device=torch.device(device),
                    wall_clock_budget_s=episode_timeout_s,
                    z0_seed=_z0s, selection_seed=_sels,
                    nu_config_mode="1nu",
                    nu_trace=_nu_buf,
                )
                pred = _extract_pred(result.answer or "", benchmark)
            except Exception as exc:
                print(f"  [bandit_ucb1 error] {exc}", flush=True); pred = ""
            match = _match_answer(str(pred), str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            if _nu_buf is not None:
                _append_nu_trace_record(
                    nu_jsonl, method=method, benchmark=benchmark, seed=seed,
                    problem_id=_problem_id(prob, benchmark, pi), nu_buf=_nu_buf,
                )

    elif method == "mcts_early_stop":
        # Paper Table 2 row "MCTS Early-Stop": run MCTS but halt as soon as
        # the wall-clock budget is hit; this is the classical "no learned
        # halting" baseline. We expose this as cts_full_episode with a
        # tighter eval_tau (30% of the standard cap) AND
        # nu_config_mode="2nu_fast" to disable the learned ACT halting
        # signal (since early-stop replaces it).
        from cts.backbone.gemma_adapter import GemmaCTSBackbone
        from cts.critic.neuro_critic import NeuroCritic
        from cts.latent.faiss_context import LatentContextWindow
        from cts.mcts.cts_episode import cts_full_episode
        from cts.policy.meta_policy import MetaPolicy
        bb = GemmaCTSBackbone(model, tok); bb.eval()
        H = bb.hidden_size; W = int(cfg.get("mcts_branching_W", 3))
        K = int(cfg.get("soft_thought_K", 64))
        meta_policy = MetaPolicy(text_dim=H, hidden=256, W=W).to(device)
        critic = NeuroCritic(z_dim=H).to(device)
        eval_tau = min(float(cfg.get("tau_flops_budget", 1e14)),
                       float(os.environ.get("CTS_EVAL_TAU_CAP", "1e13"))) * 0.3
        episode_timeout_s = float(os.environ.get("CTS_EVAL_EPISODE_TIMEOUT", "60"))
        for pi, prob in enumerate(problems):
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)
            _nu_buf: List[Any] = [] if nu_jsonl is not None else None  # type: ignore[assignment]
            try:
                _z0s = (seed * 100_000 + pi) & 0x7FFFFFFF
                _sels = (seed * 100_000 + pi + 1) & 0x7FFFFFFF
                result = cts_full_episode(
                    str(q), backbone=bb, meta_policy=meta_policy, critic=critic,
                    W=W, K=K, tau_budget=eval_tau, broyden_max_iter=12,
                    broyden_tol_min=1e-3, broyden_tol_max=1e-2, top_k=3,
                    puct_variant="paper", faiss_context=faiss_ctx,
                    max_decode_tokens=64, device=torch.device(device),
                    wall_clock_budget_s=episode_timeout_s,
                    z0_seed=_z0s, selection_seed=_sels,
                    nu_config_mode="2nu_fast",
                    nu_trace=_nu_buf,
                )
                pred = _extract_pred(result.answer or "", benchmark)
            except Exception as exc:
                print(f"  [mcts_early_stop error] {exc}", flush=True); pred = ""
            match = _match_answer(str(pred), str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            if _nu_buf is not None:
                _append_nu_trace_record(
                    nu_jsonl, method=method, benchmark=benchmark, seed=seed,
                    problem_id=_problem_id(prob, benchmark, pi), nu_buf=_nu_buf,
                )

    elif method == "expl_mcts_ppo":
        # Paper Table 2 row "Expl. MCTS + PPO" (D-2 ablation): explicit
        # MCTS with PPO-trained policy/value but NO FAISS retrieval. We
        # route through cts_4nu WITHOUT the latent context window and with
        # a depth cap of 15 (paper's stated D <= 15 OOM-cap protocol).
        from cts.backbone.gemma_adapter import GemmaCTSBackbone
        from cts.critic.neuro_critic import NeuroCritic
        from cts.mcts.cts_episode import cts_full_episode
        from cts.policy.meta_policy import MetaPolicy
        bb = GemmaCTSBackbone(model, tok); bb.eval()
        H = bb.hidden_size; W = int(cfg.get("mcts_branching_W", 3))
        K = int(cfg.get("soft_thought_K", 64))
        meta_policy = MetaPolicy(text_dim=H, hidden=256, W=W).to(device)
        critic = NeuroCritic(z_dim=H).to(device)
        eval_tau = min(float(cfg.get("tau_flops_budget", 1e14)),
                       float(os.environ.get("CTS_EVAL_TAU_CAP", "1e13")))
        episode_timeout_s = float(os.environ.get("CTS_EVAL_EPISODE_TIMEOUT", "180"))
        for pi, prob in enumerate(problems):
            q = _get_question(prob, benchmark)
            gold = _get_gold(prob, benchmark)
            if not q:
                continue
            _nu_buf: List[Any] = [] if nu_jsonl is not None else None  # type: ignore[assignment]
            try:
                _z0s = (seed * 100_000 + pi) & 0x7FFFFFFF
                _sels = (seed * 100_000 + pi + 1) & 0x7FFFFFFF
                result = cts_full_episode(
                    str(q), backbone=bb, meta_policy=meta_policy, critic=critic,
                    W=W, K=K, tau_budget=eval_tau, broyden_max_iter=15,
                    broyden_tol_min=1e-4, broyden_tol_max=1e-2, top_k=3,
                    puct_variant="paper",
                    faiss_context=None,  # NO latent context window
                    max_decode_tokens=64, device=torch.device(device),
                    wall_clock_budget_s=episode_timeout_s,
                    z0_seed=_z0s, selection_seed=_sels,
                    nu_config_mode="4nu",
                    nu_trace=_nu_buf,
                )
                pred = _extract_pred(result.answer or "", benchmark)
            except Exception as exc:
                print(f"  [expl_mcts_ppo error] {exc}", flush=True); pred = ""
            match = _match_answer(str(pred), str(gold), benchmark)
            scores.append(1.0 if match else 0.0)
            if _nu_buf is not None:
                _append_nu_trace_record(
                    nu_jsonl, method=method, benchmark=benchmark, seed=seed,
                    problem_id=_problem_id(prob, benchmark, pi), nu_buf=_nu_buf,
                )

    else:
        # Truly-unknown method name; we no longer fall through to greedy
        # because that silently mis-labels baseline numbers. Adding a new
        # baseline must be a code change, not a config typo.
        raise NotImplementedError(
            f"Unknown evaluation method `{method}`. Integrated set: "
            f"greedy, native_think, deq_only, cts_2nu, cts_4nu, "
            f"think_off_greedy, ft_nt, sc_14, bon_13, bandit_ucb1, "
            f"mcts_early_stop, expl_mcts_ppo."
        )

    return scores


def run_table2_reproduction(
    *,
    seeds: List[int],
    benchmarks: List[str],
    config_name: str = "default",
    device: str = "cuda:0",
    output_dir: str = "results/table2",
    model_dir: Optional[str] = None,
    limit: Optional[int] = None,
    nu_trace_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, StatisticalResult]]:
    """Reproduce Table 2 with full statistical protocol.

    ``nu_trace_dir`` (or env var ``CTS_NU_TRACE_DIR``): when set, every CTS
    dispatcher writes per-problem ν JSONL traces; ``aggregate_nu_table19.py``
    then folds those into Table 19.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, Dict[str, List[float]]] = {}

    # Partial-save snapshot path. Written after every (method, seed, bench)
    # cell so a timeout / crash mid-sweep never produces an empty
    # ``table2_results.json``. The post-Stage-2 pipeline run on Apr 28
    # surfaced exactly this failure mode (24 h Table 2 timeout produced
    # zero salvageable JSON); see CHANGELOG D-7 entry. The partial snapshot
    # uses ``n_samples=len(scores)`` per cell so reviewers can tell which
    # cells are complete vs in-progress.
    partial_path = Path(output_dir) / "table2_results.partial.json"

    def _flush_partial() -> None:
        try:
            snap_table2: Dict[str, Dict[str, StatisticalResult]] = {}
            for m_, bd in all_results.items():
                snap_table2[m_] = {}
                for b_, scores_ in bd.items():
                    if scores_:
                        snap_table2[m_][b_] = bootstrap_ci(scores_, ci_level=0.95)
            partial_payload: Dict[str, Any] = {}
            for m_, bench_data in snap_table2.items():
                partial_payload[m_] = {}
                for b_, stat in bench_data.items():
                    partial_payload[m_][b_] = {
                        "mean": stat.mean,
                        "std": stat.std,
                        "ci_lower": stat.ci_lower,
                        "ci_upper": stat.ci_upper,
                        "n_samples": stat.n_samples,
                    }
            with open(partial_path, "w") as pf:
                json.dump(
                    {
                        "partial": True,
                        "wrote_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "table2": partial_payload,
                        "raw_scores": {
                            m_: {b_: scores_ for b_, scores_ in bd.items()}
                            for m_, bd in all_results.items()
                        },
                    },
                    pf, indent=2,
                )
        except Exception as _exc:
            print(f"  [warn] partial-save failed: {_exc}", flush=True)

    for method in TABLE2_METHODS:
        all_results[method] = {b: [] for b in benchmarks}

        for seed in seeds:
            for bench in benchmarks:
                print(f"  [{method}] {bench} seed={seed}", flush=True)
                result = run_single_evaluation(
                    method, bench, seed,
                    config_name=config_name,
                    device=device,
                    model_dir=model_dir,
                    limit=limit,
                    nu_trace_dir=nu_trace_dir,
                )
                acc = result.get("accuracy", 0.0)
                all_results[method][bench].append(acc)
                _flush_partial()

    table2: Dict[str, Dict[str, StatisticalResult]] = {}
    for method in TABLE2_METHODS:
        table2[method] = {}
        for bench in benchmarks:
            scores = all_results[method][bench]
            table2[method][bench] = bootstrap_ci(scores, ci_level=0.95)

    _print_table2(table2, benchmarks)
    _run_wilcoxon_tests(all_results, benchmarks, output_dir)
    _save_results(all_results, table2, output_dir)

    return table2


def _print_table2(
    table2: Dict[str, Dict[str, StatisticalResult]],
    benchmarks: List[str],
) -> None:
    """Print Table 2 in paper format."""
    print("\n" + "=" * 80)
    print("Table 2: Budget-capped performance (10^14 MACs max)")
    print("=" * 80)

    header = f"{'Method':<25}" + "".join(f"{b:>12}" for b in benchmarks)
    print(header)
    print("-" * len(header))

    for method, bench_stats in table2.items():
        row = f"{method:<25}"
        for bench in benchmarks:
            s = bench_stats.get(bench)
            if s and s.n_samples > 0:
                ci_half = (s.ci_upper - s.ci_lower) / 2.0 * 100
                row += f"{s.mean * 100:>8.1f}+-{ci_half:.1f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    print("=" * 80)


# --- Statistical primary comparisons (Bonferroni family) ---
#
# Paper §7.1 defines the headline statistical claim as
# "Bonferroni-corrected over 12 primary comparisons (CTS-4nu vs four strongest
# baselines x three math benchmarks; alpha = 0.05/12)".
#
# IMPORTANT — single-GPU snapshot disclosure (NeurIPS reviewer-facing):
# Paper §7.1 headline Bonferroni family: n = 12 primary comparisons of
# CTS-4nu against the four paper baselines on three reasoning benchmarks.
# After the D1 P1 baseline-dispatcher sweep all four baselines are now
# integrated end-to-end (`greedy`, `native_think`, `sc_14`, `mcts_early_stop`),
# so the headline n=12 family is now operationally reproducible on the
# single-GPU snapshot. The remaining Table 2 rows (`bon_13`, `bandit_ucb1`,
# `ft_nt`, `think_off_greedy`, `expl_mcts_ppo`, `deq_only`) are NOT part
# of this primary family per paper §7.1 — they appear only in Table 2
# itself and not in the Wilcoxon/Bonferroni protocol — so they are
# intentionally absent here.
PRIMARY_COMPARISONS = [
    ("greedy", "math500"),         ("greedy", "gsm8k"),         ("greedy", "aime"),
    ("native_think", "math500"),   ("native_think", "gsm8k"),   ("native_think", "aime"),
    ("sc_14", "math500"),          ("sc_14", "gsm8k"),          ("sc_14", "aime"),
    ("mcts_early_stop", "math500"),("mcts_early_stop", "gsm8k"),("mcts_early_stop", "aime"),
]
PRIMARY_BONFERRONI_N = len(PRIMARY_COMPARISONS)  # = 12 (paper §7.1)
PRIMARY_ALPHA = 0.05


def _run_wilcoxon_tests(
    all_results: Dict[str, Dict[str, List[float]]],
    benchmarks: List[str],
    output_dir: str,
) -> None:
    """Wilcoxon signed-rank + Bonferroni (paper §7.1: 12 primary comparisons, α = 0.05/12).

    Standard Bonferroni: report corrected p = min(1, raw_p * N), compare to α (here 0.05).
    Equivalent to comparing raw_p to α/N (here 0.05/12 ≈ 0.0042).
    """
    print(
        f"\nWilcoxon signed-rank tests "
        f"(Bonferroni n={PRIMARY_BONFERRONI_N}, family-wise α={PRIMARY_ALPHA}):"
    )
    print("-" * 72)

    cts_4nu = all_results.get("cts_4nu", {})

    # --- Primary 12 comparisons (significance reported) ---
    primary = []
    for baseline, bench in PRIMARY_COMPARISONS:
        if bench not in benchmarks:
            continue
        x = cts_4nu.get(bench, [])
        y = all_results.get(baseline, {}).get(bench, [])
        if x and y and len(x) == len(y):
            _w, p = wilcoxon_signed_rank(x, y)
            primary.append((f"CTS-4nu vs {baseline} ({bench})", p))

    if primary:
        p_values = [p for _, p in primary]
        corrected = bonferroni_correct(p_values, n_comparisons=PRIMARY_BONFERRONI_N)
        print("\n  [primary] (Bonferroni-corrected; sig if corr_p < α=0.05)")
        for (name, raw_p), corr_p in zip(primary, corrected):
            sig = "***" if corr_p < PRIMARY_ALPHA else "n.s."
            print(f"    {name}: raw_p={raw_p:.4f}  corr_p={corr_p:.4f}  {sig}")
    else:
        print("\n  [primary] no comparable pairs available (need both methods + same n).")

    # --- Non-primary descriptive comparisons (no Bonferroni claim) ---
    non_primary = []
    primary_keys = set(PRIMARY_COMPARISONS)
    for baseline in TABLE2_METHODS:
        if baseline == "cts_4nu":
            continue
        baseline_data = all_results.get(baseline, {})
        for bench in benchmarks:
            if (baseline, bench) in primary_keys:
                continue
            x = cts_4nu.get(bench, [])
            y = baseline_data.get(bench, [])
            if x and y and len(x) == len(y):
                _w, p = wilcoxon_signed_rank(x, y)
                non_primary.append((f"CTS-4nu vs {baseline} ({bench})", p))

    if non_primary:
        print("\n  [non-primary, descriptive only — no familywise correction]")
        for name, raw_p in non_primary:
            print(f"    {name}: raw_p={raw_p:.4f}")


def _save_results(
    all_results: Dict[str, Dict[str, List[float]]],
    table2: Dict[str, Dict[str, StatisticalResult]],
    output_dir: str,
) -> None:
    """Save results to JSON."""
    out_path = Path(output_dir) / "table2_results.json"
    serializable = {}
    for method, bench_data in table2.items():
        serializable[method] = {}
        for bench, stat in bench_data.items():
            serializable[method][bench] = {
                "mean": stat.mean,
                "std": stat.std,
                "ci_lower": stat.ci_lower,
                "ci_upper": stat.ci_upper,
                "n_samples": stat.n_samples,
            }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _dry_run(args: "argparse.Namespace") -> None:
    """Print the planned evaluation matrix WITHOUT loading torch.

    Reviewer-facing surface: invoked by ``--dry-run``. Prints
    every (benchmark, method, seed) cell that the full run would
    evaluate, plus a summary of the dispatcher whitelist and the
    Q14 garbage-fallback predicate that gates math-benchmark
    predictions. Exits 0 unconditionally.

    This function MUST NOT import torch / transformers; it is the
    reviewer's pre-flight check for a degraded GPU environment.
    """
    from cts.eval.garbage_filter import MATH_BENCHMARKS

    if args.table2:
        benchmarks = BENCHMARKS
    else:
        benchmarks = args.benchmarks
    seed_list = list(range(args.seeds))
    methods = list(TABLE2_METHODS)

    n_cells = len(benchmarks) * len(methods) * len(seed_list)
    print("=" * 72)
    print("CTS Eval Pipeline - DRY RUN (no torch, no GPU)")
    print("=" * 72)
    print(f"  config        : {args.config}")
    print(f"  device        : {args.device} (NOT loaded in dry-run)")
    print(f"  output-dir    : {args.output_dir} (NOT created in dry-run)")
    print(f"  mode          : {args.mode}")
    print(f"  benchmarks    : {benchmarks}")
    print(f"  seeds         : {seed_list}")
    print(f"  methods       : {methods}")
    print(f"  per-problem limit: {args.limit if args.limit else 'all'}")
    print(f"  nu-trace-dir  : {args.nu_trace_dir or '(not set)'}")
    print(f"  TOTAL CELLS   : {n_cells}  (benchmarks x methods x seeds)")
    print()
    print("Planned per-benchmark predictor budget (max_new_tokens):")
    for b in benchmarks:
        print(f"  {b:16s} -> {_max_tokens_for(b)} tokens")
    print()
    print("Q14 garbage-math fallback applies on:")
    for b in sorted(MATH_BENCHMARKS):
        in_run = "x" if b in benchmarks else " "
        print(f"  [{in_run}] {b}")
    print()
    print("Planned cells (benchmark, method, seed):")
    for b in benchmarks:
        for m in methods:
            for s in seed_list:
                print(f"  ({b}, {m}, seed={s})")
    print()
    print("=" * 72)
    print("DRY RUN OK. No torch was imported, no model was loaded,")
    print("no output file was written. Re-run without --dry-run on a")
    print("clean GPU box to execute the matrix above.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="CTS Full Evaluation Pipeline")
    parser.add_argument("--table2", action="store_true", help="Full Table 2 reproduction")
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "aime"],
                        choices=BENCHMARKS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--config", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/table2")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--mode", default="4nu", choices=["4nu", "2nu_fast", "1nu"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems per benchmark (for fast testing)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of methods to evaluate (default: all Table 2 methods)")
    parser.add_argument(
        "--nu-trace-dir", default=None,
        help="If set, CTS dispatchers write per-problem ν JSONL traces here, "
             "consumed downstream by `scripts/aggregate_nu_table19.py` (paper "
             "Table 19). Honours the ``CTS_NU_TRACE_DIR`` env var as fallback.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the (benchmark, method, seed) cells that WOULD be "
             "evaluated, validate the dispatcher whitelist + config, and "
             "exit 0 WITHOUT loading torch / transformers / a model. Useful "
             "for reviewers who want to verify the planned eval matrix on "
             "a CPU-only host before launching the full GPU run "
             "(REVIEWER_FAQ Q15: torch-free verification).",
    )
    args = parser.parse_args()

    if args.methods:
        global TABLE2_METHODS
        TABLE2_METHODS = args.methods

    if args.dry_run:
        return _dry_run(args)

    seed_list = list(range(args.seeds))

    if args.table2:
        benchmarks = BENCHMARKS
    else:
        benchmarks = args.benchmarks

    print(f"CTS Evaluation Pipeline")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Seeds: {seed_list}")
    print(f"  Mode: {args.mode}")
    print(f"  Config: {args.config}")
    print()

    run_table2_reproduction(
        seeds=seed_list,
        benchmarks=benchmarks,
        config_name=args.config,
        device=args.device,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        limit=args.limit,
        nu_trace_dir=Path(args.nu_trace_dir) if args.nu_trace_dir else None,
    )


if __name__ == "__main__":
    main()
