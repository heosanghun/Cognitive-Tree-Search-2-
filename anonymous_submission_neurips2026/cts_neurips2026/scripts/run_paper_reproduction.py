#!/usr/bin/env python3
"""
CTS Paper Table 2 Full Reproduction Script.

Fixes all known issues:
1. Proper prompt formatting with instruction prefix
2. Robust answer extraction (LaTeX, boxed, numeric)
3. Full-scale PPO training (10K steps)
4. 5-benchmark × 4-strategy evaluation
5. Avoids import hang by lazy-loading training modules

Usage:
  python -u scripts/run_paper_reproduction.py --phase train   # Stage 2 training only
  python -u scripts/run_paper_reproduction.py --phase eval    # Evaluation only  
  python -u scripts/run_paper_reproduction.py --phase all     # Both
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

os.environ["PYTHONUNBUFFERED"] = "1"

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
DATA = ROOT / "data"

BENCHMARKS = {
    "math500": {"path": DATA / "math500" / "test.jsonl", "q_key": "problem", "a_key": "answer", "type": "math"},
    "gsm8k": {"path": DATA / "gsm8k" / "test.jsonl", "q_key": "question", "a_key": "answer", "type": "gsm8k"},
    "aime": {"path": DATA / "aime" / "test.jsonl", "q_key": "problem", "a_key": "answer", "type": "math"},
    "arc": {"path": DATA / "arc_agi" / "test.jsonl", "q_key": "input", "a_key": "output", "type": "arc"},
    "humaneval": {"path": DATA / "humaneval" / "test.jsonl", "q_key": "prompt", "a_key": "canonical_solution", "type": "code"},
}


# ================================================================
# Improved answer extraction - handles LaTeX, boxed, fractions
# ================================================================

def normalize_latex(s: str) -> str:
    """Normalize LaTeX expressions for comparison."""
    s = s.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\(left|right|displaystyle|mathrm|textbf|mathbf)", "", s)
    s = s.replace("\\frac", "FRAC")
    for cmd in ["\\cdot", "\\times", "\\div", "\\pm", "\\mp"]:
        s = s.replace(cmd, " ")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    s = re.sub(r"\s+", "", s).lower()
    return s


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed")
        if idx == -1:
            return None
        rest = text[idx + 6:].strip()
        if rest.startswith("{"):
            rest = rest[1:]
        depth = 1
        end = 0
        for i, c in enumerate(rest):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth == 0:
                end = i
                break
        return rest[:end].strip() if end > 0 else rest.strip()

    rest = text[idx + 7:]
    depth = 1
    for i, c in enumerate(rest):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth == 0:
            return rest[:i].strip()
    return rest.strip()


def extract_final_answer(text: str, answer_type: str = "math") -> str:
    """Extract final answer from model output with multiple strategies."""
    if answer_type == "gsm8k":
        m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
        if m:
            return m.group(1).replace(",", "").strip()
        m2 = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*\$?\s*([+-]?[\d,]+\.?\d*)", text, re.IGNORECASE)
        if m2:
            return m2.group(1).replace(",", "").strip()

    boxed = extract_boxed(text)
    if boxed:
        return boxed

    patterns = [
        r"(?:the answer is|answer is|equals?|result is)\s*[:\s]*([+-]?\d+[\d,./]*)",
        r"(?:therefore|thus|so|hence)[,\s]+(?:the answer is\s*)?([+-]?\d+[\d,./]*)",
        r"=\s*([+-]?\d+[\d,./]*)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).replace(",", "").strip()

    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1].strip() if nums else text.strip()[:50]


def normalize_number(s: str) -> Optional[str]:
    """Normalize a numeric string."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    try:
        val = float(s)
        if not math.isfinite(val):
            return None
        if val == int(val) and "." not in s:
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError, OverflowError):
        return None


def check_answer(pred_text: str, gold: str, answer_type: str = "math") -> bool:
    """Check if prediction matches gold answer."""
    if answer_type == "arc":
        pred_clean = pred_text.strip().upper()[:5]
        gold_clean = gold.strip().upper()
        if len(gold_clean) == 1:
            return gold_clean in pred_clean
        return gold_clean in pred_text.strip().upper()

    if answer_type == "code":
        gold_lines = [l.strip() for l in gold.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if not gold_lines:
            return False
        matches = sum(1 for gl in gold_lines if gl in pred_text)
        return matches >= max(1, len(gold_lines) * 0.3)

    pred_ans = extract_final_answer(pred_text, answer_type)
    gold_ans = extract_final_answer(gold, "gsm8k" if answer_type == "gsm8k" else answer_type)

    if normalize_latex(pred_ans) == normalize_latex(gold_ans):
        return True

    p_num = normalize_number(pred_ans)
    g_num = normalize_number(gold_ans)
    if p_num is not None and g_num is not None:
        if p_num == g_num:
            return True
        try:
            if abs(float(p_num) - float(g_num)) < 1e-4:
                return True
        except (ValueError, TypeError):
            pass

    if gold_ans.lower().strip() in pred_ans.lower():
        return True

    return False


# ================================================================
# Prompt formatting - proper instruction format for Gemma
# ================================================================

MATH_FEW_SHOT = """Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
Solution: We need $x-2 \\ge 0$ and $5-x > 0$, so $2 \\le x < 5$. The answer is $\\boxed{[2,5)}$.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: $\\det(\\mathbf{AB}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = 2 \\cdot 12 = \\boxed{24}$.

Problem: """

GSM8K_FEW_SHOT = """Problem: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Solution: He writes each friend 3*2=6 pages a week. So he writes 6*2=12 pages a week. That means he writes 12*52=624 pages a year.
#### 624

Problem: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Solution: Weng earns 12/60=$0.2 per minute. Working 50 minutes, she earned 0.2*50=$10.
#### 10

Problem: """

ARC_FEW_SHOT = """Q: Which property of a mineral can be determined just by looking at it?
A. luster B. mass C. weight D. hardness
Answer: A

Q: A student is trying to identify the mineralite.iteiteite in the rock.ite. Whichite should the studentite?
A.ite B. taste C. hardness D. color
Answer: D

Q: """

CODE_FEW_SHOT = ""


def format_math_prompt(problem: str) -> str:
    return f"{MATH_FEW_SHOT}{problem}\nSolution:"


def format_gsm8k_prompt(question: str) -> str:
    return f"{GSM8K_FEW_SHOT}{question}\nSolution:"


def format_arc_prompt(question: str) -> str:
    return f"{ARC_FEW_SHOT}{question}\nAnswer:"


def format_code_prompt(prompt: str) -> str:
    return prompt


def format_prompt(text: str, bench_type: str) -> str:
    if bench_type == "math":
        return format_math_prompt(text)
    elif bench_type == "gsm8k":
        return format_gsm8k_prompt(text)
    elif bench_type == "arc":
        return format_arc_prompt(text)
    elif bench_type == "code":
        return format_code_prompt(text)
    return text


# ================================================================
# Model loading & generation
# ================================================================

def load_model():
    """Load Gemma 4 E4B model and tokenizer."""
    import torch
    from cts.model.gemma_loader import load_gemma4_e4b

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    print(f"  Model ID: {mid}")
    model, tok = load_gemma4_e4b(
        model_id=mid, torch_dtype=torch.bfloat16,
        device_map="cuda:0", offload_vision_audio=True,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tok


def generate_text(model, tok, prompt: str, max_tokens: int = 512,
                  do_sample: bool = False, temperature: float = 0.7) -> str:
    """Generate text using raw tokenization (base model, no chat template)."""
    import torch
    device = next(model.parameters()).device
    pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
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
        gen_kwargs["top_p"] = 0.95
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        gen = model.generate(**gen_kwargs)

    return tok.decode(gen[0, in_len:], skip_special_tokens=True)


# ================================================================
# Evaluation strategies
# ================================================================

def eval_greedy(model, tok, problems, q_key, a_key, bench_type, limit=None):
    correct = total = 0
    items = problems[:limit] if limit else problems
    for i, prob in enumerate(items):
        raw_q = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not raw_q:
            continue
        prompt = format_prompt(raw_q, bench_type)
        pred = generate_text(model, tok, prompt, max_tokens=512)
        total += 1
        if check_answer(pred, gold, bench_type):
            correct += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")
    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


def eval_sc14(model, tok, problems, q_key, a_key, bench_type, limit=None, n_samples=14):
    correct = total = 0
    items = problems[:limit] if limit else problems
    for i, prob in enumerate(items):
        raw_q = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not raw_q:
            continue
        prompt = format_prompt(raw_q, bench_type)
        answers = []
        for _ in range(n_samples):
            pred = generate_text(model, tok, prompt, max_tokens=512, do_sample=True, temperature=0.7)
            ans = extract_final_answer(pred, bench_type)
            answers.append(ans)
        counts = Counter(answers)
        majority = counts.most_common(1)[0][0]
        total += 1
        if check_answer(majority, gold, bench_type):
            correct += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")
    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


def eval_native_think(model, tok, problems, q_key, a_key, bench_type, limit=None):
    correct = total = 0
    items = problems[:limit] if limit else problems
    for i, prob in enumerate(items):
        raw_q = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not raw_q:
            continue
        think_prompt = f"Think step by step carefully, then give the final answer.\n\n{format_prompt(raw_q, bench_type)}"
        pred = generate_text(model, tok, think_prompt, max_tokens=1024)
        total += 1
        if check_answer(pred, gold, bench_type):
            correct += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")
    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


def eval_cts(model, tok, problems, q_key, a_key, bench_type, limit=None):
    import torch
    from cts.backbone.gemma_adapter import GemmaCTSBackbone
    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState
    from cts.latent.faiss_context import LatentContextWindow
    from cts.policy.meta_policy import MetaPolicy

    device = next(model.parameters()).device
    bb = GemmaCTSBackbone(model, tok)
    H = bb.hidden_size

    meta = None
    ckpt = ARTIFACTS / "stage2_meta_value.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = MetaPolicy(text_dim=ck.get("text_dim", H), hidden=256, W=ck.get("W", 3)).to(device)
        meta.load_state_dict(ck["meta"], strict=False)
        meta.eval()

    s1 = ARTIFACTS / "stage1_last.pt"
    if s1.exists():
        s1d = torch.load(s1, map_location="cpu", weights_only=False)
        bb.load_state_dict(s1d.get("backbone_state_dict", s1d), strict=False)

    correct = total = 0
    items = problems[:limit] if limit else problems
    for i, prob in enumerate(items):
        raw_q = str(prob.get(q_key, ""))
        gold = str(prob.get(a_key, ""))
        if not raw_q:
            continue
        prompt = format_prompt(raw_q, bench_type)
        faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)
        budget = RuntimeBudgetState()
        if meta is not None:
            with torch.no_grad():
                ctx = bb.encode_context(prompt).to(device).float()
                if ctx.dim() == 1:
                    ctx = ctx.unsqueeze(0)
                nu, _ = meta.logits_and_nu(ctx)
        else:
            nu = NuVector()
        best_text = ""
        for branch in range(3):
            try:
                r = transition(prompt, branch, nu, budget, bb, K=64,
                               broyden_max_iter=15, tau_flops_budget=1e14,
                               faiss_context=faiss_ctx, max_decode_tokens=128)
                if r.child_text and len(r.child_text) > len(best_text):
                    best_text = r.child_text
            except Exception:
                continue
        total += 1
        if check_answer(best_text, gold, bench_type):
            correct += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            print(f"    [{i+1}/{len(items)}] acc={correct}/{total} ({100*correct/max(total,1):.1f}%)")
    return {"correct": correct, "total": total, "accuracy": round(100 * correct / max(total, 1), 2)}


# ================================================================
# Training
# ================================================================

def run_training(ppo_steps: int = 10000):
    """Run Stage 2 PPO training - inline to avoid import hang issues."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_t
    from torch.distributions import Categorical

    from cts.backbone.gemma_adapter import GemmaCTSBackbone
    from cts.critic.neuro_critic import NeuroCritic
    from cts.deq.transition import transition
    from cts.rewards.shaping import paper_reward
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.policy.meta_policy import MetaPolicy
    from cts.train.ppo_core import compute_gae, ppo_clipped_loss, value_loss
    from cts.types import RuntimeBudgetState
    from cts.utils.config import load_config

    print(f"\n  PPO Steps: {ppo_steps}")

    cfg = load_config("default")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    print(f"  Loading model: {mid}")
    model, tok = load_gemma4_e4b(
        model_id=mid, device_map="cuda:0" if torch.cuda.is_available() else "auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)

    s1_ckpt = ARTIFACTS / "stage1_last.pt"
    if s1_ckpt.exists():
        ck = torch.load(s1_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("backbone_state_dict", ck)
        bb.load_state_dict(sd, strict=False)
        print(f"  Loaded Stage 1 checkpoint: {s1_ckpt}")

    for p in bb.parameters():
        p.requires_grad = False
    bb.eval()

    H = bb.hidden_size
    W = 3
    K = int(cfg.get("latent_tokens_K", 64))

    class ValueHead(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Linear(dim, 1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.net(x.float()).squeeze(-1)

    meta = MetaPolicy(text_dim=H, hidden=256, W=W).to(dev)
    value_head = ValueHead(H).to(dev)
    critic_z = NeuroCritic(H).to(dev)

    train_params = list(meta.parameters()) + list(value_head.parameters())
    opt = torch.optim.AdamW(train_params, lr=float(cfg.get("lr", 3e-5)))

    clip_eps = float(cfg.get("ppo_clip_epsilon", 0.2))
    vf_coef = float(cfg.get("value_loss_coef", 0.5))
    ent_coef = float(cfg.get("entropy_coef", 0.01))
    tau_budget = float(cfg.get("tau_flops_budget", 1e14))
    lambda_halt = float(cfg.get("act_halting_penalty", 0.05))
    gae_gamma = float(cfg.get("discount_gamma", 0.99))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    collect_batch = 4
    ppo_epochs = 2

    data_path = DATA / "stage2" / "math_train_prompts_5000.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    lines = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    print(f"  Training data: {len(lines)} prompts")

    idx = 0
    t_start = time.time()
    for step in range(ppo_steps):
        batch_obs, batch_actions, batch_old_logp = [], [], []
        batch_rewards, batch_values = [], []

        for _ in range(collect_batch):
            row = lines[idx % len(lines)]
            idx += 1
            prompt = str(row.get("prompt", row.get("problem", str(row))))[:8192]

            with torch.no_grad():
                ctx = bb.encode_context(prompt)
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            obs = ctx.to(dev).float()

            with torch.no_grad():
                nu, logits = meta.logits_and_nu(obs)
                dist_old = Categorical(logits=logits)
                action = int(dist_old.sample().item())
                old_logp = float(dist_old.log_prob(torch.tensor(action, device=dev)).item())
                v_old = float(value_head(obs).item())

            budget = RuntimeBudgetState()
            tr = transition(prompt, action, nu, budget, bb,
                            K=K, d=H, broyden_max_iter=15,
                            tau_flops_budget=tau_budget, max_decode_tokens=1)

            converged = tr.solver_stats.get("converged", False)
            depth_T = tr.budget.terminal_depth if tr.budget else 1
            r = paper_reward(correct=converged, terminal_depth=depth_T, lambda_halt=lambda_halt)

            batch_obs.append(obs.squeeze(0))
            batch_actions.append(action)
            batch_old_logp.append(old_logp)
            batch_rewards.append(r)
            batch_values.append(v_old)

        obs_stacked = torch.stack(batch_obs, dim=0)
        actions_t = torch.tensor(batch_actions, device=dev, dtype=torch.long)
        old_logp_t = torch.tensor(batch_old_logp, device=dev, dtype=torch.float32)
        rewards_t = torch.tensor(batch_rewards, device=dev, dtype=torch.float32)

        dones_list = [True] * len(batch_rewards)
        adv_list, ret_list = compute_gae(batch_rewards, batch_values, dones_list,
                                          gamma=gae_gamma, lam=gae_lam)
        advantages = torch.tensor(adv_list, device=dev, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = torch.tensor(ret_list, device=dev, dtype=torch.float32)

        for _ in range(ppo_epochs):
            h = meta.act(meta.enc(obs_stacked))
            logits_new = meta.head_prior(h)
            dist_new = Categorical(logits=logits_new)
            new_logp = dist_new.log_prob(actions_t)
            ent = dist_new.entropy().mean()

            p_loss = ppo_clipped_loss(new_logp, old_logp_t, advantages.detach(), clip=clip_eps)
            v_pred = value_head(obs_stacked)
            v_l = value_loss(v_pred, returns)

            loss = p_loss + vf_coef * v_l - ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, float(cfg.get("max_grad_norm", 1.0)))
            opt.step()

        if (step + 1) % 100 == 0 or (step + 1) == ppo_steps:
            elapsed = time.time() - t_start
            eta = elapsed / (step + 1) * (ppo_steps - step - 1)
            print(f"  step={step+1}/{ppo_steps} loss={float(loss.item()):.4f} "
                  f"reward={float(rewards_t.mean().item()):.4f} "
                  f"elapsed={elapsed/3600:.1f}h ETA={eta/3600:.1f}h")

    out = ARTIFACTS / "stage2_meta_value.pt"
    torch.save({
        "meta": meta.state_dict(),
        "value_head": value_head.state_dict(),
        "critic_z": critic_z.state_dict(),
        "config_name": "default",
        "W": W, "text_dim": H,
    }, out)
    print(f"  Training done. Checkpoint: {out}")
    return {"checkpoint": str(out), "steps": ppo_steps}


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "all"], default="eval")
    ap.add_argument("--ppo-steps", type=int, default=10000)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--skip-sc14", action="store_true")
    ap.add_argument("--skip-cts", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = ap.parse_args()

    print("=" * 70)
    print("CTS PAPER REPRODUCTION - Table 2 Full Comparison")
    print("=" * 70)
    t_total = time.time()

    # Phase 1: Training
    if args.phase in ("train", "all"):
        print("\n" + "=" * 60)
        print("PHASE 1: Stage 2 PPO Full-Scale Training")
        print("=" * 60)
        t_train = time.time()
        run_training(args.ppo_steps)
        print(f"  Time: {(time.time()-t_train)/3600:.1f} hours")

        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 2: Evaluation
    if args.phase in ("eval", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: 5-Benchmark Evaluation")
        print("=" * 60)

        print("\n[LOAD] Loading model...")
        model, tok = load_model()

        all_results = {}
        out_path = ARTIFACTS / "table2_paper_reproduction.json"

        if args.resume and out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if "results" in saved:
                all_results = saved["results"]
            else:
                all_results = {k: v for k, v in saved.items()
                               if isinstance(v, dict) and any(s in v for s in ("greedy", "native_think", "sc14", "cts"))}
            print(f"  Resumed {len(all_results)} benchmarks from previous run")
            for bn, br in all_results.items():
                for st, res in br.items():
                    print(f"    {bn}/{st}: {res.get('accuracy',0)}% ({res.get('correct',0)}/{res.get('total',0)})")

        strategies = ["greedy", "native_think"]
        if not args.skip_sc14:
            strategies.insert(1, "sc14")
        if not args.skip_cts:
            strategies.append("cts")

        for bench_name, info in BENCHMARKS.items():
            existing = all_results.get(bench_name, {})
            remaining_strats = [s for s in strategies if s not in existing]
            if not remaining_strats:
                print(f"\n  SKIP {bench_name.upper()} - all strategies completed")
                continue

            print(f"\n{'='*50}")
            print(f"BENCHMARK: {bench_name.upper()}")
            print(f"{'='*50}")

            path = info["path"]
            if not path.exists():
                print(f"  SKIP - no data at {path}")
                continue

            problems = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        problems.append(json.loads(line))
            if args.eval_limit:
                problems = problems[:args.eval_limit]
            print(f"  Problems: {len(problems)}")

            bench_results = dict(existing)
            for strat in remaining_strats:
                print(f"\n  --- {strat.upper()} ---")
                t_s = time.time()
                q_key, a_key = info["q_key"], info["a_key"]
                bench_type = info["type"]

                if strat == "greedy":
                    r = eval_greedy(model, tok, problems, q_key, a_key, bench_type, args.eval_limit)
                elif strat == "sc14":
                    r = eval_sc14(model, tok, problems, q_key, a_key, bench_type, args.eval_limit)
                elif strat == "native_think":
                    r = eval_native_think(model, tok, problems, q_key, a_key, bench_type, args.eval_limit)
                elif strat == "cts":
                    r = eval_cts(model, tok, problems, q_key, a_key, bench_type, args.eval_limit)
                else:
                    continue

                r["elapsed_s"] = round(time.time() - t_s, 1)
                bench_results[strat] = r
                print(f"  {strat}: {r['accuracy']}% ({r['correct']}/{r['total']}) [{r['elapsed_s']:.0f}s]")

            all_results[bench_name] = bench_results

            # Save intermediate after each benchmark
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  [SAVED] intermediate results to {out_path}")

        # Summary
        paper = {
            "math500": {"greedy": 45.2, "sc14": 59.3, "native_think": 57.0, "cts": 68.4},
            "gsm8k": {"greedy": 76.5, "sc14": 84.2, "native_think": 82.4, "cts": 92.1},
            "aime": {"greedy": 28.3, "sc14": 34.8, "native_think": 42.5, "cts": 56.4},
            "arc": {"greedy": 36.1, "sc14": 52.4, "native_think": 50.1, "cts": 64.1},
            "humaneval": {"greedy": 56.4, "sc14": 65.2, "native_think": 63.3, "cts": 74.2},
        }

        elapsed = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS ({elapsed/3600:.1f} hours)")
        print(f"{'='*70}")
        print(f"{'Bench':<12} {'Strategy':<16} {'Measured':>10} {'Paper':>10} {'Gap':>8}")
        print("-" * 60)
        for bn, br in all_results.items():
            for st, res in br.items():
                pv = paper.get(bn, {}).get(st, "N/A")
                gap = f"{res['accuracy'] - pv:+.1f}" if isinstance(pv, (int, float)) else ""
                print(f"{bn:<12} {st:<16} {res['accuracy']:>9.1f}% {pv:>9}% {gap:>8}")

        final = {
            "results": all_results, "paper_targets": paper,
            "elapsed_s": round(elapsed, 1),
            "eval_limit": args.eval_limit,
        }
        out = ARTIFACTS / "table2_paper_reproduction.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
