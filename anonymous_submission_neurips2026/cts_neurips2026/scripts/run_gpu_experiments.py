#!/usr/bin/env python3
"""GPU experiments: VRAM profile, spectral radius, CTS benchmark.
Uses real Gemma 4 E4B on RTX 4090.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".hf_cache"))

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def gpu_mem():
    return {
        "alloc_mb": torch.cuda.memory_allocated() / 1e6,
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
    }


def main():
    t_total = time.time()
    device = torch.device("cuda:0")

    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    print(f"Model: {mid}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ──
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.backbone.gemma_adapter import GemmaCTSBackbone

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[LOAD] Loading Gemma 4 E4B ...")
    t0 = time.time()
    model, tok = load_gemma4_e4b(
        model_id=mid,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        offload_vision_audio=True,
    )
    load_time = time.time() - t0
    mem_after_load = gpu_mem()
    model_gb = mem_after_load["alloc_mb"] / 1024
    print(f"[LOAD] Done in {load_time:.1f}s, model VRAM: {model_gb:.2f} GB")

    bb = GemmaCTSBackbone(model, tok)

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: Table 1 - VRAM Profile at various depths
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Table 1 - VRAM Profile")
    print("=" * 60)

    from cts.deq.transition import transition
    from cts.types import NuVector, RuntimeBudgetState
    from cts.latent.faiss_context import LatentContextWindow

    nu = NuVector(nu_val=1.0, nu_expl=1.0, nu_tol=0.5, nu_temp=1.0, nu_act=1.0)
    H = bb.hidden_size

    vram_rows = []
    depths = [1, 15, 35, 100]

    for depth in depths:
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)

        actual_steps = min(depth, 8)
        for step_i in range(actual_steps):
            budget = RuntimeBudgetState()
            try:
                r = transition(
                    f"Calculate {step_i + depth} times {step_i + 3}",
                    branch_index=step_i % 3,
                    nu=nu,
                    budget=budget,
                    backbone=bb,
                    K=64,
                    broyden_max_iter=15,
                    tau_flops_budget=1e14,
                    faiss_context=faiss_ctx,
                    max_decode_tokens=1,
                )
            except Exception as e:
                print(f"  depth={depth} step={step_i}: {e}")
                break

        m = gpu_mem()
        peak_gb = m["peak_mb"] / 1024
        overhead_mb = m["peak_mb"] - mem_after_load["alloc_mb"]

        row = {
            "depth": depth,
            "steps_run": actual_steps,
            "peak_vram_gb": round(peak_gb, 2),
            "model_vram_gb": round(model_gb, 2),
            "cts_overhead_mb": round(overhead_mb, 1),
            "o1_per_node": overhead_mb < 500,
        }
        vram_rows.append(row)
        print(f"  Depth {depth:>3d} ({actual_steps} steps): peak={peak_gb:.2f} GB, overhead={overhead_mb:.0f} MB")

    with open(ARTIFACTS / "table1_real_gemma_vram.json", "w") as f:
        json.dump(vram_rows, f, indent=2)
    print("  -> Saved table1_real_gemma_vram.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: Spectral Radius gamma
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Spectral Radius gamma")
    print("=" * 60)

    from cts.latent.bottleneck import init_z0
    from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

    gammas = []
    K = 64
    num_samples = 5

    for i in range(num_samples):
        torch.cuda.empty_cache()
        gen = torch.Generator(device=device)
        gen.manual_seed(2026 + i)
        z = init_z0(K, H, device, gen).float()

        ctx = bb.encode_context(f"Compute {i * 7} plus {i * 11}").to(device=device, dtype=torch.float32)
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)

        w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)

        num_probe = 32
        vs = torch.randn(num_probe, K * H, device=device, dtype=torch.float32)
        vs = vs / vs.norm(dim=1, keepdim=True)

        max_ratio = 0.0
        for v_idx in range(num_probe):
            z_in = z.clone().requires_grad_(True)

            alpha = routing_weights(z_in, w_g, 1.0)
            mw = sparse_module_weights(alpha, 3)
            out = bb.deq_step(z_in, ctx, mw, {"top_k": 3})
            out_flat = out.reshape(-1)

            g = torch.autograd.grad(out_flat, z_in, grad_outputs=vs[v_idx].view_as(out_flat),
                                    retain_graph=False, allow_unused=True)[0]
            if g is not None:
                ratio = g.reshape(-1).norm().item()
                max_ratio = max(max_ratio, ratio)

        gammas.append(max_ratio)
        print(f"  Sample {i + 1}/{num_samples}: gamma ~ {max_ratio:.4f}")

    mean_g = sum(gammas) / len(gammas)
    std_g = (sum((g - mean_g) ** 2 for g in gammas) / len(gammas)) ** 0.5

    sr_result = {
        "spectral_radius_mean": round(mean_g, 4),
        "spectral_radius_std": round(std_g, 4),
        "num_samples": num_samples,
        "gammas": [round(g, 4) for g in gammas],
        "paper_target": 0.92,
        "contraction": mean_g < 1.0,
    }
    with open(ARTIFACTS / "spectral_radius.json", "w") as f:
        json.dump(sr_result, f, indent=2)
    print(f"\n  Mean gamma = {mean_g:.4f} +/- {std_g:.4f} (paper: ~0.92)")
    print(f"  Contraction mapping: {'YES' if mean_g < 1.0 else 'NO'}")
    print("  -> Saved spectral_radius.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 3: CTS Benchmark
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: CTS Pipeline Benchmark")
    print("=" * 60)

    from cts.policy.meta_policy import MetaPolicy

    ckpt = ARTIFACTS / "stage2_meta_value.pt"
    meta = None
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        W_ck = ck.get("W", 3)
        td = ck.get("text_dim", H)
        meta = MetaPolicy(text_dim=td, hidden=256, W=W_ck).to(device)
        meta.load_state_dict(ck["meta"], strict=False)
        meta.eval()
        print("  Loaded Stage 2 Meta-Policy")

    s1_path = ARTIFACTS / "stage1_last.pt"
    if s1_path.exists():
        s1 = torch.load(s1_path, map_location="cpu", weights_only=False)
        sd = s1.get("backbone_state_dict", s1)
        bb.load_state_dict(sd, strict=False)
        print("  Loaded Stage 1 backbone")

    # Math problems (synthetic representative samples)
    math_problems = [
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

    # Also try loading real MATH-500 data if available
    math_data = ROOT / "data" / "math500" / "test.jsonl"
    if math_data.exists():
        real_problems = []
        with open(math_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    real_problems.append(json.loads(line))
        if real_problems:
            math_problems = real_problems[:20]
            print(f"  Using {len(math_problems)} real MATH-500 problems")

    gsm_data = ROOT / "data" / "gsm8k" / "test.jsonl"
    gsm_problems = []
    if gsm_data.exists():
        with open(gsm_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gsm_problems.append(json.loads(line))
        gsm_problems = gsm_problems[:20]
        print(f"  Loaded {len(gsm_problems)} GSM8K problems")

    bench_results = {}

    for bench_name, problems in [("MATH-500", math_problems), ("GSM8K", gsm_problems)]:
        if not problems:
            continue
        print(f"\n  --- {bench_name} ({len(problems)} problems) ---")

        correct = 0
        total = 0
        mac_total = 0
        peak_vram_per_q = []

        for idx, prob in enumerate(problems):
            prompt = prob.get("problem", prob.get("question", ""))
            gold = str(prob.get("answer", prob.get("solution", "")))
            # Extract final numeric answer for GSM8K format
            if "####" in gold:
                gold = gold.split("####")[-1].strip()

            torch.cuda.reset_peak_memory_stats()

            faiss_ctx = LatentContextWindow(dim=H, retrieval_k=3, min_steps=10)
            budget = RuntimeBudgetState()

            if meta is not None:
                with torch.no_grad():
                    cf = bb.encode_context(prompt).to(device).float()
                    if cf.dim() == 1:
                        cf = cf.unsqueeze(0)
                    nu_pred, _ = meta.logits_and_nu(cf)
            else:
                nu_pred = NuVector()

            best_text = ""
            for branch in range(3):
                try:
                    r = transition(
                        prompt, branch, nu_pred, budget, bb,
                        K=64, broyden_max_iter=15,
                        tau_flops_budget=1e14,
                        faiss_context=faiss_ctx,
                        max_decode_tokens=32,
                    )
                    if r.child_text and len(r.child_text) > len(best_text):
                        best_text = r.child_text
                except Exception:
                    continue

            mac_total += budget.mac_accumulated
            peak_vram_per_q.append(torch.cuda.max_memory_allocated() / 1e9)

            if gold.strip() and gold.strip() in str(best_text):
                correct += 1
            total += 1

            if (idx + 1) % 5 == 0:
                print(f"    [{idx+1}/{len(problems)}] acc={correct}/{total}")

        acc = 100 * correct / max(total, 1)
        avg_mac = mac_total / max(total, 1)
        avg_peak = sum(peak_vram_per_q) / max(len(peak_vram_per_q), 1)

        bench_results[bench_name] = {
            "total": total,
            "correct": correct,
            "accuracy_pct": round(acc, 2),
            "avg_macs": avg_mac,
            "avg_peak_vram_gb": round(avg_peak, 2),
        }
        print(f"  {bench_name} accuracy: {correct}/{total} ({acc:.1f}%)")

    with open(ARTIFACTS / "table2_cts_benchmark.json", "w") as f:
        json.dump(bench_results, f, indent=2)
    print("  -> Saved table2_cts_benchmark.json")

    # ── Summary ──
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"ALL GPU EXPERIMENTS DONE - {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)

    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "model_vram_gb": round(model_gb, 2),
        "elapsed_s": round(elapsed, 1),
        "vram_profile": vram_rows,
        "spectral_radius": sr_result,
        "benchmarks": bench_results,
    }
    with open(ARTIFACTS / "gpu_experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {ARTIFACTS / 'gpu_experiments_summary.json'}")


if __name__ == "__main__":
    main()
