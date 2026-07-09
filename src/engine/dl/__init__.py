"""MIE-DL — deep sequence model on raw intraday bars (pre-registered spec).

Torch is an optional dependency: the base engine and its tests must run
without it. Every module in this package that needs torch calls
``require_torch()`` first, which raises with the exact install command.
"""
from __future__ import annotations


def require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MIE-DL needs PyTorch (CPU). Install with:\n"
            '  pip install "torch>=2.6,<2.8" --index-url '
            "https://download.pytorch.org/whl/cpu"
        ) from exc
