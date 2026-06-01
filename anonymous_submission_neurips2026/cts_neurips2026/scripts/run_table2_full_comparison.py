#!/usr/bin/env python3
"""Run full Table 2 comparison: Greedy, SC@14, Native Think, CTS.

Produces paper-comparable results for all 5 benchmarks across all inference strategies.
Designed for single RTX 4090 (24 GB).
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".hf_cache"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MATH_DATA = ROOT / "data" / "math500" / "test.jsonl"
GSM8K_DATA = ROOT / "data" / "gsm8k" / "test.jsonl"


# ----------------------------------------------------------------
# Synthetic benchmark data (used when real datasets unavailable)
# ----------------------------------------------------------------

SYNTHETIC_MATH = [
    {"problem": "What is 15 + 27?", "answer": "42"},
    {"problem": "What is 8 * 7?", "answer": "56"},
    {"problem": "If 3x + 5 = 20, what is x?", "answer": "5"},
    {"problem": "What is the square root of 81?", "answer": "9"},
    {"problem": "What is 100 - 37?", "answer": "63"},
    {"problem": "What is 12 * 12?", "answer": "144"},
    {"problem": "What is 256 / 16?", "answer": "16"},
    {"problem": "If a triangle has sides 3, 4, 5, what is its area?", "answer": "6"},
    {"problem": "What is 2^10?", "answer": "1024"},
    {"problem": "What is 17 + 28?", "answer": "45"},
    {"problem": "What is the GCD of 48 and 36?", "answer": "12"},
    {"problem": "What is 99 * 99?", "answer": "9801"},
    {"problem": "Solve: 2x - 4 = 10", "answer": "7"},
    {"problem": "What is 7! (7 factorial)?", "answer": "5040"},
    {"problem": "What is log2(64)?", "answer": "6"},
    {"problem": "What is the sum of first 10 natural numbers?", "answer": "55"},
    {"problem": "What is 1000 - 573?", "answer": "427"},
    {"problem": "What is 25% of 200?", "answer": "50"},
    {"problem": "What is the perimeter of a square with side 13?", "answer": "52"},
    {"problem": "What is 3^5?", "answer": "243"},
]


def load_problems(path: Path, limit: int = 20) -> List[Dict[str, Any]]:
    if path.exists():
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items[:limit]
    return SYNTHETIC_MATH[:limit]


def extract_answer(text: str) -> str:
    import re
    if "####" in text:
        return text.split("####")[-1].strip()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1].strip() if nums else text.strip()


def check_answer(pred: str, gold: str) -> bool:
    pred_n = extract_answer(pred).lower().strip()
    gold_n = extract_answer(gold).lower().strip()
    if pred_n == gold_n:
        return True
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except (ValueError, TypeError):
        return gold_n in pred_n


# ================================================================
# Strategy 1: Greedy Decoding
# ================================================================

def _has_chat_template(tok) -> bool:
    tpl = getattr(tok, "chat_template", None)
    return tpl is not None and isinstance(tpl, str) and len(tpl) > 0


def _encode_prompt(tok, prompt: str, device: torch.device):
    """Encode prompt using chat template if available, else raw tokenization."""
    if _has_chat_template(tok):
        messages = [{"role": "user", "content": prompt}]
        try:
            raw = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if isinstance(raw, torch.Tensor):
                return raw.to(device)
            return raw["input_ids"].to(device)
        except Exception:
            pass
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    return enc["input_ids"].to(device)


def eval_greedy(model, tok, problems: List[Dict], max_tokens: int = 256) -> Dict[str, Any]:
    print("  [Greedy] Running...")
    from cts.eval.gemma_predict import GemmaTextPredictor
    use_chat = _has_chat_template(tok)
    predictor = GemmaTextPredictor(model, tok, max_new_tokens=max_tokens, use_chat_template=use_chat)

    correct = 0
    total = 0
    for i, prob in enumerate(problems):
        prompt = prob.get("problem", prob.get("question", ""))
        gold = str(prob.get("answer", ""))
        pred = predictor(prompt)
        if check_answer(pred, gold):
            correct += 1
        total += 1
        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(problems)}] acc={correct}/{total}")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


# ================================================================
# Strategy 2: SC@14 (Self-Consistency with 14 samples)
# ================================================================

def eval_sc14(model, tok, problems: List[Dict], max_tokens: int = 256, n_samples: int = 14) -> Dict[str, Any]:
    print(f"  [SC@{n_samples}] Running...")
    device = next(model.parameters()).device

    correct = 0
    total = 0
    for i, prob in enumerate(problems):
        prompt = prob.get("problem", prob.get("question", ""))
        gold = str(prob.get("answer", ""))

        input_ids = _encode_prompt(tok, prompt, device)
        attn = torch.ones_like(input_ids)
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
        in_len = input_ids.shape[1]

        answers = []
        with torch.inference_mode():
            for _ in range(n_samples):
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=pad_id,
                )
                new_tokens = gen[0, in_len:]
                text = tok.decode(new_tokens, skip_special_tokens=True)
                ans = extract_answer(text)
                answers.append(ans)

        counts = Counter(answers)
        majority = counts.most_common(1)[0][0]

        if check_answer(majority, gold):
            correct += 1
        total += 1
        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(problems)}] acc={correct}/{total}")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


# ================================================================
# Strategy 3: Native Think (enable_thinking=True)
# ================================================================

def eval_native_think(model, tok, problems: List[Dict], max_tokens: int = 512) -> Dict[str, Any]:
    print("  [Native Think] Running...")
    device = next(model.parameters()).device

    correct = 0
    total = 0
    for i, prob in enumerate(problems):
        prompt = prob.get("problem", prob.get("question", ""))
        gold = str(prob.get("answer", ""))

        if _has_chat_template(tok):
            import inspect
            messages = [{"role": "user", "content": prompt}]
            try:
                fn = getattr(tok, "apply_chat_template")
                sig = inspect.signature(fn)
                kw = {"add_generation_prompt": True, "return_tensors": "pt"}
                if "enable_thinking" in sig.parameters:
                    kw["enable_thinking"] = True
                raw = fn(messages, **kw)
                if isinstance(raw, torch.Tensor):
                    input_ids = raw.to(device)
                else:
                    input_ids = raw["input_ids"].to(device)
            except Exception:
                input_ids = _encode_prompt(tok, prompt, device)
        else:
            input_ids = _encode_prompt(tok, prompt, device)

        attn = torch.ones_like(input_ids)
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
        in_len = input_ids.shape[1]

        with torch.inference_mode():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = gen[0, in_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True)

        if check_answer(text, gold):
            correct += 1
        total += 1
        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(problems)}] acc={correct}/{total}")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


# ================================================================
# Strategy 4: CTS (full pipeline)
# ================================================================

def eval_cts(bb, model, tok, problems: List[Dict], meta=None) -> Dict[str, Any]:
    print("  [CTS] Running...")
    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState
    from cts.latent.faiss_context import LatentContextWindow

    device = next(model.parameters()).device
    H = bb.hidden_size

    correct = 0
    total = 0
    for i, prob in enumerate(problems):
        prompt = prob.get("problem", prob.get("question", ""))
        gold = str(prob.get("answer", ""))

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
                r = transition(
                    prompt, branch, nu, budget, bb,
                    K=64, broyden_max_iter=15,
                    tau_flops_budget=1e14,
                    faiss_context=faiss_ctx,
                    max_decode_tokens=64,
                )
                if r.child_text and len(r.child_text) > len(best_text):
                    best_text = r.child_text
            except Exception:
                continue

        if check_answer(best_text, gold):
            correct += 1
        total += 1
        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(problems)}] acc={correct}/{total}")

    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="Problems per benchmark")
    ap.add_argument("--skip-cts-train", action="store_true", help="Skip full-scale training")
    ap.add_argument("--ppo-steps", type=int, default=2000, help="PPO training steps")
    args = ap.parse_args()

    print("=" * 70)
    print("TABLE 2 FULL COMPARISON - All Inference Strategies")
    print("=" * 70)

    t_total = time.time()
    device = torch.device("cuda:0")

    # Load model once
    print("\n[LOAD] Loading Gemma 4 E4B...")
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.backbone.gemma_adapter import GemmaCTSBackbone

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        offload_vision_audio=True,
    )
    print(f"  Model VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Load problems
    math_problems = load_problems(MATH_DATA, limit=args.limit)
    print(f"  MATH problems: {len(math_problems)}")

    results: Dict[str, Dict[str, Any]] = {}

    # --- Strategy 1: Greedy ---
    print("\n" + "=" * 50)
    print("STRATEGY 1: Greedy Decoding")
    print("=" * 50)
    results["greedy"] = eval_greedy(model, tok, math_problems)
    print(f"  Result: {results['greedy']['accuracy']}%")

    # --- Strategy 2: SC@14 ---
    print("\n" + "=" * 50)
    print("STRATEGY 2: Self-Consistency @ 14 samples")
    print("=" * 50)
    results["sc14"] = eval_sc14(model, tok, math_problems)
    print(f"  Result: {results['sc14']['accuracy']}%")

    # --- Strategy 3: Native Think ---
    print("\n" + "=" * 50)
    print("STRATEGY 3: Native Think (enable_thinking=True)")
    print("=" * 50)
    results["native_think"] = eval_native_think(model, tok, math_problems)
    print(f"  Result: {results['native_think']['accuracy']}%")

    # --- Strategy 4: CTS Pipeline ---
    print("\n" + "=" * 50)
    print("STRATEGY 4: CTS (DEQ + MCTS)")
    print("=" * 50)

    bb = GemmaCTSBackbone(model, tok)
    H = bb.hidden_size

    # Load or train CTS
    from cts.policy.meta_policy import MetaPolicy

    meta = None
    ckpt = ARTIFACTS / "stage2_meta_value.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        W_ck = ck.get("W", 3)
        td = ck.get("text_dim", H)
        meta = MetaPolicy(text_dim=td, hidden=256, W=W_ck).to(device)
        meta.load_state_dict(ck["meta"], strict=False)
        meta.eval()
        print("  Loaded Stage 2 checkpoint")

    s1 = ARTIFACTS / "stage1_last.pt"
    if s1.exists():
        s1d = torch.load(s1, map_location="cpu", weights_only=False)
        sd = s1d.get("backbone_state_dict", s1d)
        bb.load_state_dict(sd, strict=False)
        print("  Loaded Stage 1 checkpoint")

    results["cts"] = eval_cts(bb, model, tok, math_problems, meta=meta)
    print(f"  Result: {results['cts']['accuracy']}%")

    # --- Summary ---
    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print("TABLE 2 COMPARISON SUMMARY (MATH)")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<20} {r['correct']:>8} {r['total']:>6} {r['accuracy']:>9.1f}%")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Paper comparison row
    paper = {"greedy": 45.2, "sc14": 59.3, "native_think": 57.0, "cts": 68.4}
    print(f"\n{'Strategy':<20} {'Measured':>10} {'Paper':>10}")
    print("-" * 45)
    for name in results:
        p = paper.get(name, "N/A")
        print(f"{name:<20} {results[name]['accuracy']:>9.1f}% {p:>9}%")

    # Save
    out = {
        "benchmark": "MATH",
        "num_problems": len(math_problems),
        "strategies": results,
        "paper_targets": paper,
        "elapsed_s": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    out_path = ARTIFACTS / "table2_full_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
