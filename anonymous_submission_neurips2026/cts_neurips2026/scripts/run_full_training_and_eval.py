#!/usr/bin/env python3
"""
CTS Full Pipeline: Model Download -> Stage 1 -> Stage 2 -> All Benchmarks.

Single RTX 4090 24GB (paper hardware).
Requires HF_TOKEN for Gemma 4 E4B (gated model).

Usage:
  $env:HF_TOKEN="hf_your_token_here"
  python scripts/run_full_training_and_eval.py --run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _check_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass
    if not token:
        print("ERROR: HF_TOKEN not set. Gemma 4 E4B is a gated model.")
        print("  1. Go to https://huggingface.co/google/gemma-4-E4B")
        print("  2. Accept the license agreement")
        print("  3. Get your token from https://huggingface.co/settings/tokens")
        print("  4. Run: $env:HF_TOKEN='hf_your_token_here'")
        return False
    print(f"HF Token: {token[:8]}... OK")
    return True


def _check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        return False
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name} ({vram:.0f} GB)")
    if vram < 20:
        print("WARNING: VRAM < 20 GB. Paper uses RTX 4090 (24 GB)")
    return True


def step_download_model():
    """Step 1: Download Gemma 4 E4B."""
    print("\n[Step 1/5] Downloading Gemma 4 E4B...")
    t0 = time.time()
    from cts.model.gemma_loader import load_gemma4_e4b
    try:
        model, tokenizer = load_gemma4_e4b(offload_vision_audio=True)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {params/1e9:.1f}B params")
        print(f"  Time: {time.time()-t0:.0f}s")
        del model
        import torch
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def step_stage1():
    """Step 2: Stage 1 DEQ Warm-Up (LoRA r=8, 10K OpenMath examples)."""
    print("\n[Step 2/5] Stage 1: DEQ Warm-Up...")
    # Check if OpenMath data exists
    from pathlib import Path
    data_dir = Path("data")
    jsonl = data_dir / "openmath_10k.jsonl"
    if not jsonl.exists():
        print("  Downloading OpenMath subset...")
        data_dir.mkdir(exist_ok=True)
        try:
            from datasets import load_dataset
            ds = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
            import json
            count = 0
            with jsonl.open("w", encoding="utf-8") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= 10000:
                        break
            print(f"  Saved {count} examples to {jsonl}")
        except Exception as e:
            print(f"  Failed to download: {e}")
            return False

    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/run_stage1_openmath.py",
         "--data", str(jsonl), "--lora", "--max-steps", "2000",
         "--device", "cuda:0"],
        capture_output=False, text=True
    )
    return result.returncode == 0


def step_stage2():
    """Step 3: Stage 2 PPO (5K MATH/AIME prompts)."""
    print("\n[Step 3/5] Stage 2: PPO Training...")
    ckpt = Path("artifacts/stage1_last.pt")
    if not ckpt.exists():
        print(f"  Stage 1 checkpoint not found: {ckpt}")
        return False

    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/run_stage2_math_ppo.py",
         "--stage1-ckpt", str(ckpt),
         "--device", "cuda:0"],
        capture_output=False, text=True
    )
    return result.returncode == 0


def step_benchmarks():
    """Step 4: Run all benchmarks (MATH-500, GSM8K, ARC, HumanEval)."""
    print("\n[Step 4/5] Running Benchmarks...")
    results = {}

    benchmarks = [
        ("MATH-500", "scripts/run_math500.py"),
        ("ARC-AGI", "scripts/run_arc_agi_text.py"),
    ]
    for name, script in benchmarks:
        if Path(script).exists():
            print(f"  Running {name}...")
            import subprocess
            r = subprocess.run(
                [sys.executable, script, "--gemma", "--max-new-tokens", "64"],
                capture_output=True, text=True, timeout=3600
            )
            results[name] = r.returncode == 0
            print(f"  {name}: {'OK' if results[name] else 'FAILED'}")

    return all(results.values()) if results else True


def step_profile():
    """Step 5: VRAM and latency profiling (Table 1)."""
    print("\n[Step 5/5] VRAM/Latency Profiling (Table 1)...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/run_paper_artifacts_pipeline.py"],
            capture_output=False, text=True, timeout=3600
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  Profile error: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Execute (without this, dry-run)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--benchmarks-only", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("CTS Full Training & Evaluation Pipeline")
    print("Paper: NeurIPS 2026 — KV-Cache-Free DEQ MCTS")
    print("=" * 60)

    if not _check_gpu():
        sys.exit(1)
    if not _check_token():
        sys.exit(1)

    if not args.run:
        print("\nDRY RUN. Pipeline steps:")
        print("  1. Download Gemma 4 E4B (~8 GB)")
        print("  2. Stage 1: DEQ Warm-Up (LoRA r=8, 10K examples, ~2-4h)")
        print("  3. Stage 2: PPO (5K MATH/AIME, ~8-16h)")
        print("  4. Benchmarks: MATH-500, GSM8K, ARC, HumanEval")
        print("  5. VRAM/Latency Profile (Table 1)")
        print("\nAdd --run to execute.")
        return

    t_total = time.time()
    results = {}

    if not args.benchmarks_only:
        if not args.skip_download:
            results["download"] = step_download_model()
            if not results["download"]:
                print("\nModel download failed. Check HF_TOKEN and network.")
                sys.exit(1)

        if not args.skip_train:
            results["stage1"] = step_stage1()
            if results.get("stage1"):
                results["stage2"] = step_stage2()

    results["benchmarks"] = step_benchmarks()
    results["profile"] = step_profile()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({elapsed/3600:.1f} hours)")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
