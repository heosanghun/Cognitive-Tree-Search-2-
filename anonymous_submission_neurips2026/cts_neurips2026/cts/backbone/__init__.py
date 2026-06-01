"""Backbone registry with PEP-562 lazy loading.

The Gemma adapter pulls in ``transformers`` + ``torch_compile`` and ~6 GB of
weights at first instantiation; importing it eagerly at package-load time
dominates ``import cts`` startup and makes CPU-only reviewer machines crash
on torch_compile probes. We therefore expose ``GemmaCTSBackbone`` (and any
other heavy backbones) via :pep:`562` ``__getattr__`` so users only pay the
cost when they actually do ``cts.backbone.GemmaCTSBackbone(...)`` or
``from cts.backbone import GemmaCTSBackbone``.

Lightweight backbones (the protocol class and the mock backbone used for
unit tests) are imported eagerly because they have no native dependencies.
"""

from __future__ import annotations

from typing import Any

from cts.backbone.protocol import BaseCTSBackbone
from cts.backbone.mock_tiny import MockTinyBackbone

__all__ = ["BaseCTSBackbone", "MockTinyBackbone", "GemmaCTSBackbone"]

_LAZY = {
    "GemmaCTSBackbone": ("cts.backbone.gemma_adapter", "GemmaCTSBackbone"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY:
        import importlib

        mod_path, attr = _LAZY[name]
        module = importlib.import_module(mod_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'cts.backbone' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY.keys()))
