"""Paper Table 3: Gemma 42 layers -> 19 modules (code uses 0-based layer_idx)."""

from __future__ import annotations

from typing import Dict, List

# module index 0..18 corresponds to m1..m19 in the paper.


def _build_layer_to_module() -> Dict[int, int]:
    m: Dict[int, int] = {}
    # m1-m4 : paper layers 1-8  -> idx 0-7, 2 layers per module
    for i in range(0, 8):
        m[i] = i // 2
    # m5-m8 : paper layers 9-16 -> idx 8-15
    for i in range(8, 16):
        m[i] = 4 + (i - 8) // 2
    # m9-m14 : paper layers 17-28 -> idx 16-27, 6 modules, 2 layers each
    for i in range(16, 28):
        m[i] = 8 + (i - 16) // 2
    # m15-m19 : paper layers 29-42 -> idx 28-41, 5 modules, 14 layers
    for i in range(28, 42):
        off = i - 28
        if off < 3:
            mod = 14
        elif off < 6:
            mod = 15
        elif off < 9:
            mod = 16
        elif off < 12:
            mod = 17
        else:
            mod = 18
        m[i] = mod
    return m


LAYER_TO_MODULE: Dict[int, int] = _build_layer_to_module()


def module_for_layer(layer_idx: int) -> int:
    if layer_idx < 0 or layer_idx > 41:
        raise IndexError(f"layer_idx {layer_idx} out of 0..41")
    return LAYER_TO_MODULE[layer_idx]


def layers_for_module(module_idx: int) -> List[int]:
    return [l for l, mo in LAYER_TO_MODULE.items() if mo == module_idx]
