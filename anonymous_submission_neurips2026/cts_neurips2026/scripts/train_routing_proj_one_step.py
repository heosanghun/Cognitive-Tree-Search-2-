#!/usr/bin/env python3
"""
One Adam step on `routing_proj` (W_g) — mock (no weights) or Gemma (full load).

  # No checkpoint — instant
  python scripts/train_routing_proj_one_step.py --mock

  # Full Gemma 4 E4B + GemmaCTSBackbone (blend DEQ map recommended)
  set CTS_GEMMA_MODEL_DIR=<repo_root>\gemma-4-E4B-it
  set CTS_DEQ_MAP_MODE=blend
  python scripts/train_routing_proj_one_step.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.routing_proj_step import MockRoutingOnly, train_routing_proj_one_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="19x64 routing_proj only (no Gemma)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device-map", type=str, default=None)
    ap.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Add entropy_coef * H(alpha) to routing loss (paper-style trade-off knob)",
    )
    args = ap.parse_args()

    if args.mock:
        bb = MockRoutingOnly(d=64)
        loss, _ = train_routing_proj_one_step(bb, lr=args.lr, entropy_coef=args.entropy_coef)
        print("mock_loss=", loss)
        return

    dm = args.device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok = load_gemma4_e4b(device_map=dm, torch_dtype=torch.bfloat16)
    bb = GemmaCTSBackbone(model, tok)
    loss, _ = train_routing_proj_one_step(bb, lr=args.lr, entropy_coef=args.entropy_coef)
    print("routing_proj_one_step_loss=", loss)


if __name__ == "__main__":
    main()
