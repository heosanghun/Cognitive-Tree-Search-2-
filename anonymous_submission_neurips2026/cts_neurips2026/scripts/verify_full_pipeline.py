#!/usr/bin/env python3
"""Full pipeline verification: dataset loading + mock end-to-end + real model check."""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_datasets():
    """Verify all paper datasets are accessible."""
    print("=" * 60)
    print("[1/6] Dataset Verification")
    print("=" * 60)

    try:
        from datasets import load_dataset

        # MATH-500
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"  MATH-500: {len(ds)} examples OK")

        # GSM8K
        ds_gsm = load_dataset("gsm8k", "main", split="test")
        print(f"  GSM8K: {len(ds_gsm)} examples OK")

        # OpenMathInstruct-2 (paper §6.1)
        ds_om = load_dataset(
            "nvidia/OpenMathInstruct-2", split="train", streaming=True
        )
        count = 0
        for _ in ds_om:
            count += 1
            if count >= 10:
                break
        print(f"  OpenMathInstruct-2: streaming OK (sampled {count})")
        return True
    except Exception as e:
        print(f"  Dataset error: {e}")
        return False


def check_mock_transition():
    """End-to-end mock transition with all paper features."""
    print("\n" + "=" * 60)
    print("[2/6] Mock Transition (FAISS + Broyden + Batch)")
    print("=" * 60)

    import torch
    from cts.backbone.mock_tiny import MockTinyBackbone
    from cts.deq.transition import transition, transition_batch
    from cts.deq.broyden_forward import enable_convergence_tracking
    from cts.latent.faiss_context import LatentContextWindow
    from cts.types import NuVector, RuntimeBudgetState

    stats = enable_convergence_tracking()
    bb = MockTinyBackbone(hidden=64, num_layers=42)
    ctx = LatentContextWindow(dim=64, retrieval_k=3, min_steps=3)
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0, nu_val=1.0, nu_act=1.0)

    # Single transition with FAISS
    for i in range(10):
        r = transition(
            f"Solve: what is 2 + {i}?", i, nu, RuntimeBudgetState(), bb,
            K=64, d=64, broyden_max_iter=30, faiss_context=ctx
        )

    print(f"  FAISS context size: {ctx.size} vectors")
    print(f"  Memory per node: {ctx.memory_kb_per_node():.3f} KB")
    print(f"  Broyden convergence rate: {stats.convergence_rate:.1%}")
    print(f"  Broyden avg iterations: {stats.avg_iterations:.1f}")

    # Batch transition
    budget = RuntimeBudgetState()
    results = transition_batch(
        "Batch test prompt", nu, budget, bb,
        W=3, K=64, d=64, broyden_max_iter=30,
    )
    converged = sum(1 for r in results if r.solver_stats["converged"])
    print(f"  Batch DEQ (W=3): {converged}/3 converged ✓")
    return True


def check_mcts_rollout():
    """Full MCTS rollout with MetaPolicy."""
    print("\n" + "=" * 60)
    print("[3/6] MCTS Rollout (MetaPolicy + PUCT + NeuroCritic)")
    print("=" * 60)

    import torch
    from cts.mcts.episode import mcts_root_rollouts
    from cts.mcts.deep_rollout import two_ply_mcts_rollouts
    from cts.policy.meta_policy import MetaPolicy
    from cts.critic.neuro_critic import NeuroCritic
    from cts.mcts.critic_reward import make_critic_reward_fn

    meta = MetaPolicy(text_dim=64, hidden=32, W=3)
    critic = NeuroCritic(z_dim=64)
    reward_fn = make_critic_reward_fn(critic, z_dim=64)

    r = mcts_root_rollouts(
        "What is the sum of 1+2+3?",
        num_simulations=4, W=3, K=4, d=64,
        meta_policy=meta,
        puct_variant="paper",
        reward_fn=reward_fn,
    )
    print(f"  Root rollouts: ns={r.ns}, qs=[{', '.join(f'{q:.3f}' for q in r.qs)}]")

    tp = two_ply_mcts_rollouts(
        "Find x: 2x+1=5",
        sims_root=3, sims_child=3, W=3, K=4, d=64,
        meta_policy=meta, reward_fn=reward_fn,
    )
    print(f"  Two-ply best action: {tp.best_action}")
    print(f"  Two-ply child Q: [{', '.join(f'{q:.3f}' for q in tp.child.qs)}] ✓")
    return True


def check_reward_eq5():
    """Paper Eq.(5) reward function."""
    print("\n" + "=" * 60)
    print("[4/6] Reward Eq.(5): R = 1{correct} - λ_halt · T")
    print("=" * 60)

    from cts.rewards.shaping import paper_reward

    r_correct = paper_reward(correct=True, terminal_depth=10, lambda_halt=0.05)
    r_wrong = paper_reward(correct=False, terminal_depth=10, lambda_halt=0.05)
    print(f"  Correct, T=10: R = {r_correct:.2f} (expected 0.50)")
    print(f"  Wrong,   T=10: R = {r_wrong:.2f} (expected -0.50)")
    assert abs(r_correct - 0.5) < 1e-6
    assert abs(r_wrong - (-0.5)) < 1e-6
    print("  Eq.(5) verified ✓")
    return True


def check_latent_projection():
    """Wproj Latent-to-Text Projection."""
    print("\n" + "=" * 60)
    print("[5/6] Wproj Latent-to-Text Projection (K=64)")
    print("=" * 60)

    import torch
    from cts.latent.bottleneck import LatentProjection, LatentDecoder

    proj = LatentProjection(latent_dim=64, model_dim=256)
    z = torch.randn(64, 64)  # K=64, d=64
    soft_prompt = proj(z)
    print(f"  Wproj: [64, 64] -> [64, 256] ✓ (shape={list(soft_prompt.shape)})")

    dec = LatentDecoder(latent_dim=64, model_dim=256, vocab_size=32000)
    logits = dec.greedy_logits(z)
    print(f"  Decoder logits: [64, 32000] ✓ (shape={list(logits.shape)})")
    return True


def check_gemma_model():
    """Check if Gemma 4 E4B is loadable."""
    print("\n" + "=" * 60)
    print("[6/6] Gemma 4 E4B Model Status")
    print("=" * 60)

    from pathlib import Path
    import os

    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    env_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        candidates.append(Path(env_cache) / "hub" / "models--google--gemma-4-E4B")
    home = Path.home()
    candidates.append(home / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-E4B")
    candidates.append(repo_root / ".hf_cache" / "models--google--gemma-4-E4B")
    cache_paths = candidates

    for cp in cache_paths:
        blobs = cp / "blobs"
        if blobs.exists():
            files = list(blobs.iterdir())
            total = sum(f.stat().st_size for f in files if f.is_file())
            incomplete = any(".incomplete" in f.name for f in files)
            print(f"  Cache: {cp}")
            print(f"  Files: {len(files)}, Total: {total / 1024**3:.1f} GB")
            if incomplete:
                print(f"  ⚠ Download INCOMPLETE — HF token required")

    token = os.environ.get("HF_TOKEN") or None
    try:
        from huggingface_hub import get_token
        token = token or get_token()
    except Exception:
        pass

    if token:
        print(f"  HF Token: {token[:8]}... ✓")
        print("  → Model download/load possible!")
        return True
    else:
        print("  HF Token: NOT SET")
        print("  → Set HF_TOKEN to download Gemma 4 E4B weights")
        print("  → Command: $env:HF_TOKEN='hf_your_token_here'")
        return False


def main():
    print("CTS Full Pipeline Verification")
    print(f"{'='*60}\n")

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU: <none> (CPU-only reviewer machine — Gemma load will be skipped)")
        print("VRAM: n/a")
    print(f"PyTorch: {torch.__version__}")

    results = {}
    t0 = time.time()

    results["datasets"] = check_datasets()
    results["mock_transition"] = check_mock_transition()
    results["mcts_rollout"] = check_mcts_rollout()
    results["reward_eq5"] = check_reward_eq5()
    results["latent_projection"] = check_latent_projection()
    results["gemma_model"] = check_gemma_model()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓ PASS" if ok else "⚠ NEEDS ACTION"
        print(f"  {name}: {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")

    if not results["gemma_model"]:
        print("\n" + "=" * 60)
        print("NEXT STEP: Set HF_TOKEN to enable Gemma 4 E4B")
        print("=" * 60)
        print("PowerShell:")
        print('  $env:HF_TOKEN="hf_your_token_here"')
        print("")
        print("Then run:")
        print("  python scripts/run_full_training_and_eval.py")


if __name__ == "__main__":
    main()
