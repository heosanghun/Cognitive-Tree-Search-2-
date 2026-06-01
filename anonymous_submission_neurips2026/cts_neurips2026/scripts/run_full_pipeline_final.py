#!/usr/bin/env python3
"""
CTS Full Pipeline: Stage 2 Full-Scale Training + 5 Benchmark Evaluation.

Phase 1: Stage 2 PPO training (10K steps)
Phase 2: 5-benchmark × 4-strategy evaluation with real datasets

Hardware: Single RTX 4090 (24 GB)
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
DATA = ROOT / "data"

BENCHMARKS = {
    "math500": {"path": DATA / "math500" / "test.jsonl", "q_key": "problem", "a_key": "answer"},
    "gsm8k": {"path": DATA / "gsm8k" / "test.jsonl", "q_key": "question", "a_key": "answer"},
    "aime": {"path": DATA / "aime" / "test.jsonl", "q_key": "problem", "a_key": "answer"},
    "arc": {"path": DATA / "arc_agi" / "test.jsonl", "q_key": "input", "a_key": "output"},
    "humaneval": {"path": DATA / "humaneval" / "test.jsonl", "q_key": "prompt", "a_key": "canonical_solution"},
}

# ================================================================
# Answer extraction and comparison
# ================================================================

def extract_math_answer(text: str) -> str:
    """Extract final answer from model output."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    the_answer = re.search(r"(?:the answer is|answer is|= )\s*([+-]?\d+\.?\d*)", text, re.IGNORECASE)
    if the_answer:
        return the_answer.group(1).strip()
    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1].strip() if nums else text.strip()[:50]


def extract_gsm8k_gold(answer_text: str) -> str:
    """Extract gold answer from GSM8K format (#### N)."""
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    nums = re.findall(r"[+-]?\d+\.?\d*", answer_text)
    return nums[-1] if nums else answer_text.strip()


def normalize_number(s: str) -> Optional[str]:
    """Normalize a number string for comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return None


def check_math_answer(pred: str, gold: str) -> bool:
    """Compare two math answers with normalization."""
    pred_ans = extract_math_answer(pred)
    gold_ans = extract_math_answer(gold)

    if pred_ans.lower().strip() == gold_ans.lower().strip():
        return True

    p_num = normalize_number(pred_ans)
    g_num = normalize_number(gold_ans)
    if p_num is not None and g_num is not None and p_num == g_num:
        return True

    return gold_ans.lower().strip() in pred_ans.lower()


def check_arc_answer(pred: str, gold: str) -> bool:
    """Compare ARC answers (single letter or text)."""
    pred_c = pred.strip().upper()
    gold_c = gold.strip().upper()
    if len(gold_c) == 1:
        return pred_c.startswith(gold_c) or gold_c in pred_c[:5]
    return pred_c == gold_c


def check_humaneval(pred: str, gold: str) -> bool:
    """Compare HumanEval: check if key parts of canonical solution appear."""
    gold_lines = [l.strip() for l in gold.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    if not gold_lines:
        return False
    matches = sum(1 for gl in gold_lines if gl in pred)
    return matches >= len(gold_lines) * 0.5


# ================================================================
# Data loading
# ================================================================

def load_benchmark(name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load benchmark data."""
    info = BENCHMARKS[name]
    path = info["path"]
    if not path.exists():
        print(f"  WARNING: {path} not found!")
        return []

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if limit:
        items = items[:limit]
    return items


# ================================================================
# Inference strategies
# ================================================================

def _has_chat_template(tok) -> bool:
    tpl = getattr(tok, "chat_template", None)
    return tpl is not None and isinstance(tpl, str) and len(tpl) > 0


def _generate_text(model, tok, prompt: str, device, max_tokens: int = 512,
                   do_sample: bool = False, temperature: float = 1.0) -> str:
    """Generate text from prompt."""
    if _has_chat_template(tok):
        try:
            messages = [{"role": "user", "content": prompt}]
            raw = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if isinstance(raw, torch.Tensor):
                input_ids = raw.to(device)
            else:
                input_ids = raw["input_ids"].to(device)
        except Exception:
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = enc["input_ids"].to(device)
    else:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)

    attn = torch.ones_like(input_ids)
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
    in_len = input_ids.shape[1]

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_tokens,
        "pad_token_id": pad_id,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        gen = model.generate(**gen_kwargs)

    new_tokens = gen[0, in_len:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def eval_greedy(model, tok, problems, q_key, a_key, bench_name, device, limit=None) -> Dict:
    """Strategy 1: Greedy decoding."""
    correct = total = 0
    items = problems[:limit] if limit else problems
    for i, prob in enumerate(items):
        prompt = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not prompt:
            continue

        pred = _generate_text(model, tok, prompt, device, max_tokens=512)
        total += 1

        if bench_name == "gsm8k":
            gold_num = extract_gsm8k_gold(gold)
            match = check_math_answer(pred, gold_num)
        elif bench_name == "humaneval":
            match = check_humaneval(pred, gold)
        elif bench_name == "arc":
            match = check_arc_answer(pred, gold)
        else:
            match = check_math_answer(pred, gold)

        if match:
            correct += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


def eval_sc14(model, tok, problems, q_key, a_key, bench_name, device, limit=None, n_samples=14) -> Dict:
    """Strategy 2: Self-Consistency @ 14 samples."""
    correct = total = 0
    items = problems[:limit] if limit else problems
    for i, prob in enumerate(items):
        prompt = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not prompt:
            continue

        answers = []
        for _ in range(n_samples):
            pred = _generate_text(model, tok, prompt, device, max_tokens=512,
                                  do_sample=True, temperature=0.7)
            ans = extract_math_answer(pred)
            answers.append(ans)

        counts = Counter(answers)
        majority = counts.most_common(1)[0][0]
        total += 1

        if bench_name == "gsm8k":
            gold_num = extract_gsm8k_gold(gold)
            match = check_math_answer(majority, gold_num)
        elif bench_name == "humaneval":
            match = check_humaneval(majority, gold)
        elif bench_name == "arc":
            match = check_arc_answer(majority, gold)
        else:
            match = check_math_answer(majority, gold)

        if match:
            correct += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


def eval_native_think(model, tok, problems, q_key, a_key, bench_name, device, limit=None) -> Dict:
    """Strategy 3: Native Think (enable_thinking=True)."""
    correct = total = 0
    items = problems[:limit] if limit else problems

    import inspect
    has_thinking = False
    if _has_chat_template(tok):
        try:
            fn = getattr(tok, "apply_chat_template")
            sig = inspect.signature(fn)
            has_thinking = "enable_thinking" in sig.parameters
        except Exception:
            pass

    for i, prob in enumerate(items):
        prompt = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not prompt:
            continue

        if has_thinking:
            messages = [{"role": "user", "content": prompt}]
            try:
                raw = tok.apply_chat_template(messages, add_generation_prompt=True,
                                               return_tensors="pt", enable_thinking=True)
                if isinstance(raw, torch.Tensor):
                    input_ids = raw.to(device)
                else:
                    input_ids = raw["input_ids"].to(device)
                attn = torch.ones_like(input_ids)
                pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
                in_len = input_ids.shape[1]
                with torch.inference_mode():
                    gen = model.generate(input_ids=input_ids, attention_mask=attn,
                                         max_new_tokens=1024, do_sample=False, pad_token_id=pad_id)
                pred = tok.decode(gen[0, in_len:], skip_special_tokens=True)
            except Exception:
                pred = _generate_text(model, tok, prompt, device, max_tokens=1024)
        else:
            think_prompt = f"Think step by step, then give the final answer.\n\n{prompt}"
            pred = _generate_text(model, tok, think_prompt, device, max_tokens=1024)

        total += 1

        if bench_name == "gsm8k":
            gold_num = extract_gsm8k_gold(gold)
            match = check_math_answer(pred, gold_num)
        elif bench_name == "humaneval":
            match = check_humaneval(pred, gold)
        elif bench_name == "arc":
            match = check_arc_answer(pred, gold)
        else:
            match = check_math_answer(pred, gold)

        if match:
            correct += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


def eval_cts(bb, model, tok, problems, q_key, a_key, bench_name, device, meta=None, limit=None) -> Dict:
    """Strategy 4: CTS (DEQ + MCTS)."""
    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState
    from cts.latent.faiss_context import LatentContextWindow

    H = bb.hidden_size
    correct = total = 0
    items = problems[:limit] if limit else problems

    for i, prob in enumerate(items):
        prompt = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not prompt:
            continue

        faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)
        budget = RuntimeBudgetState()

        if meta is not None:
            with torch.no_grad():
                ctx_feat = bb.encode_context(prompt).to(device).float()
                if ctx_feat.dim() == 1:
                    ctx_feat = ctx_feat.unsqueeze(0)
                nu, _ = meta.logits_and_nu(ctx_feat)
        else:
            nu = NuVector()

        best_text = ""
        for branch in range(3):
            try:
                r = transition(prompt, branch, nu, budget, bb,
                               K=64, broyden_max_iter=15, tau_flops_budget=1e14,
                               faiss_context=faiss_ctx, max_decode_tokens=64)
                if r.child_text and len(r.child_text) > len(best_text):
                    best_text = r.child_text
            except Exception:
                continue

        total += 1

        if bench_name == "gsm8k":
            gold_num = extract_gsm8k_gold(gold)
            match = check_math_answer(best_text, gold_num)
        elif bench_name == "humaneval":
            match = check_humaneval(best_text, gold)
        elif bench_name == "arc":
            match = check_arc_answer(best_text, gold)
        else:
            match = check_math_answer(best_text, gold)

        if match:
            correct += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-steps", type=int, default=10000)
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="Limit problems per benchmark (None=full)")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-sc14", action="store_true", help="Skip SC@14 (slowest)")
    ap.add_argument("--skip-cts", action="store_true", help="Skip CTS eval")
    args = ap.parse_args()

    print("=" * 70)
    print("CTS FULL PIPELINE: Training + 5-Benchmark Evaluation")
    print("=" * 70)
    t_total = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Phase 1: Full-scale Training ----
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("PHASE 1: Stage 2 PPO Full-Scale Training")
        print("=" * 70)

        t_train = time.time()
        from cts.train.stage2_ppo_train import run_stage2_math_ppo

        s1_ckpt = ARTIFACTS / "stage1_last.pt"
        s1_path = str(s1_ckpt) if s1_ckpt.exists() else None
        if s1_path:
            print(f"  Using Stage 1 checkpoint: {s1_path}")

        data_path = DATA / "stage2" / "math_train_prompts_5000.jsonl"
        print(f"  Training data: {data_path}")
        print(f"  PPO Steps: {args.ppo_steps}")
        print(f"  Starting training...")

        result = run_stage2_math_ppo(
            math_prompts_jsonl=str(data_path),
            config_name="default",
            total_steps=args.ppo_steps,
            device="cuda:0",
            W=3,
            K=64,
            collect_batch=4,
            ppo_epochs=2,
            broyden_max_iter=15,
            parallel_map=False,
            stage1_checkpoint=s1_path,
            use_critic_reward=False,
            log_every=50,
        )
        print(f"  Training completed in {(time.time()-t_train)/3600:.1f} hours")
        print(f"  Checkpoint: {result['checkpoint']}")

        del run_stage2_math_ppo
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Phase 2: 5-Benchmark Evaluation ----
    print("\n" + "=" * 70)
    print("PHASE 2: 5-Benchmark × 4-Strategy Evaluation")
    print("=" * 70)

    # Load model
    print("\n[LOAD] Loading Gemma 4 E4B...")
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.backbone.gemma_adapter import GemmaCTSBackbone

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid, torch_dtype=torch.bfloat16,
        device_map="cuda:0", offload_vision_audio=True,
    )
    print(f"  Model VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load CTS components
    bb = GemmaCTSBackbone(model, tok)
    H = bb.hidden_size

    meta = None
    ckpt = ARTIFACTS / "stage2_meta_value.pt"
    if ckpt.exists():
        from cts.policy.meta_policy import MetaPolicy
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        W_ck = ck.get("W", 3)
        td = ck.get("text_dim", H)
        meta = MetaPolicy(text_dim=td, hidden=256, W=W_ck).to(device)
        meta.load_state_dict(ck["meta"], strict=False)
        meta.eval()
        print("  Loaded Stage 2 checkpoint for CTS eval")

    s1 = ARTIFACTS / "stage1_last.pt"
    if s1.exists():
        s1d = torch.load(s1, map_location="cpu", weights_only=False)
        sd = s1d.get("backbone_state_dict", s1d)
        bb.load_state_dict(sd, strict=False)
        print("  Loaded Stage 1 backbone checkpoint")

    # Evaluate each benchmark
    all_results: Dict[str, Dict[str, Any]] = {}

    strategies = ["greedy", "native_think"]
    if not args.skip_sc14:
        strategies.insert(1, "sc14")
    if not args.skip_cts:
        strategies.append("cts")

    for bench_name, bench_info in BENCHMARKS.items():
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bench_name.upper()}")
        print(f"{'='*60}")

        problems = load_benchmark(bench_name, limit=args.eval_limit)
        if not problems:
            print(f"  Skipping - no data")
            continue

        print(f"  Problems: {len(problems)}")
        q_key = bench_info["q_key"]
        a_key = bench_info["a_key"]
        bench_results: Dict[str, Any] = {}

        for strat in strategies:
            print(f"\n  --- {strat.upper()} ---")
            t_s = time.time()

            if strat == "greedy":
                r = eval_greedy(model, tok, problems, q_key, a_key, bench_name, device, args.eval_limit)
            elif strat == "sc14":
                r = eval_sc14(model, tok, problems, q_key, a_key, bench_name, device, args.eval_limit)
            elif strat == "native_think":
                r = eval_native_think(model, tok, problems, q_key, a_key, bench_name, device, args.eval_limit)
            elif strat == "cts":
                r = eval_cts(bb, model, tok, problems, q_key, a_key, bench_name, device, meta, args.eval_limit)
            else:
                continue

            elapsed = time.time() - t_s
            r["elapsed_s"] = round(elapsed, 1)
            bench_results[strat] = r
            print(f"  {strat}: {r['accuracy']}% ({r['correct']}/{r['total']}) [{elapsed:.0f}s]")

        all_results[bench_name] = bench_results

        # Save intermediate results
        out_path = ARTIFACTS / "table2_real_benchmark_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    total_elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print("FINAL RESULTS: Table 2 Comparison (Real Datasets)")
    print("=" * 70)

    paper_targets = {
        "math500": {"greedy": 45.2, "sc14": 59.3, "native_think": 57.0, "cts": 68.4},
        "gsm8k": {"greedy": 76.5, "sc14": 84.2, "native_think": 82.4, "cts": 92.1},
        "aime": {"greedy": 28.3, "sc14": 34.8, "native_think": 42.5, "cts": 56.4},
        "arc": {"greedy": 36.1, "sc14": 52.4, "native_think": 50.1, "cts": 64.1},
        "humaneval": {"greedy": 56.4, "sc14": 65.2, "native_think": 63.3, "cts": 74.2},
    }

    print(f"\n{'Benchmark':<12} {'Strategy':<16} {'Measured':>10} {'Paper':>10} {'Gap':>8}")
    print("-" * 60)
    for bench_name, bench_res in all_results.items():
        for strat, res in bench_res.items():
            paper_val = paper_targets.get(bench_name, {}).get(strat, "N/A")
            gap = ""
            if isinstance(paper_val, (int, float)):
                gap = f"{res['accuracy'] - paper_val:+.1f}"
            print(f"{bench_name:<12} {strat:<16} {res['accuracy']:>9.1f}% {paper_val:>9}% {gap:>8}")
        print()

    print(f"\nTotal pipeline time: {total_elapsed/3600:.1f} hours")

    # Save final results
    final = {
        "results": all_results,
        "paper_targets": paper_targets,
        "total_elapsed_s": round(total_elapsed, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "eval_limit": args.eval_limit,
        "ppo_steps": args.ppo_steps,
        "skip_training": args.skip_training,
    }
    out_path = ARTIFACTS / "table2_real_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
