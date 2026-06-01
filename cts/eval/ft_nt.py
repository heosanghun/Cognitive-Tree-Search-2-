"""Fine-tuned + Native Think (FT-NT) predictor builder (paper Table 2).

Loads Stage-1 LoRA weights into a Gemma backbone and returns a callable
predictor that uses the native-think chat template path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.eval.gemma_predict import GemmaTextPredictor
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.lora_compat import apply_paper_lora


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_ft_nt_predictor(
    *,
    stage1_ckpt: Path | str,
    device: str = "cuda:0",
    model_dir: Optional[str] = None,
    lora_rank: int = 8,
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj", "o_proj"),
    max_new_tokens: int = 512,
) -> tuple[Callable[..., str], GemmaCTSBackbone, Any]:
    """Return ``(predictor, backbone, tokenizer)`` with Stage-1 LoRA loaded.

    The predictor accepts ``temperature`` / ``do_sample`` kwargs (forwarded
    to ``GemmaTextPredictor``) for parity with SC@14 / BoN@13 dispatchers.
    """
    ck_path = Path(stage1_ckpt)
    if not ck_path.is_file():
        raise FileNotFoundError(f"FT-NT stage1 checkpoint missing: {ck_path}")

    import os

    mid = model_dir or os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=device,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)
    ck = _load_torch(ck_path)
    sd = ck.get("backbone_state_dict", ck)
    if any(k.endswith("lora_A.weight") or k.endswith("lora_B.weight") for k in sd):
        apply_paper_lora(
            bb,
            rank=lora_rank,
            target_modules=lora_targets,
            dropout=0.05,
            require_match=True,
            verbose=True,
        )
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    if missing:
        print(
            f"[ft_nt] load_state_dict missing keys (head 3/{len(missing)}): "
            f"{missing[:3]}",
            flush=True,
        )
    if unexpected:
        print(
            f"[ft_nt] load_state_dict unexpected keys (head 3/{len(unexpected)}): "
            f"{unexpected[:3]}",
            flush=True,
        )
    bb.eval()
    predictor = GemmaTextPredictor(
        bb.cg,
        tok,
        max_new_tokens=max_new_tokens,
        device=device,
        use_chat_template=True,
    )
    return predictor, bb, tok
