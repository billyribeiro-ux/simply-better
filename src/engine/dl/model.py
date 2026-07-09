"""CausalTCN — dual-branch dilated temporal convolutional network (~96k params).

Architecture is FIXED A PRIORI (pre-registered 2026-07-09, ledger); no search
may be run against the 2024-2026 range. Convolutions are left-padded so output
t sees inputs <= t only (causality also holds trivially because sequences end
at the decision bar). Heads: 3-class (down/flat/up = top/neither/bottom) and
return quantiles (q10, q50, q90).
"""
from __future__ import annotations

from . import require_torch

torch = require_torch()
import torch.nn as nn  # noqa: E402


class _CausalBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 3, p_drop: float = 0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, ch, k, dilation=dilation))
        self.c2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, ch, k, dilation=dilation))
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def _cconv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return conv(nn.functional.pad(x, (self.pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self._cconv(self.c1, x)))
        h = self.drop(self.act(self._cconv(self.c2, h)))
        return x + h


class _Branch(nn.Module):
    def __init__(self, in_ch: int, ch: int, dilations: list[int]):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.ModuleList([_CausalBlock(ch, d) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h[:, :, -1]                       # last timestep


class CausalTCN(nn.Module):
    N_SYMBOLS = 8

    def __init__(self, ch1: int = 48, ch5: int = 32, emb: int = 8):
        super().__init__()
        self.branch1 = _Branch(9, ch1, [1, 2, 4, 8, 16])
        self.branch5 = _Branch(6, ch5, [1, 2, 4])
        self.emb = nn.Embedding(self.N_SYMBOLS, emb)
        d = ch1 + ch5 + emb + 2
        self.trunk = nn.Sequential(nn.Linear(d, 64), nn.GELU())
        self.head_cls = nn.Linear(64, 3)
        self.head_q = nn.Linear(64, 3)

    def forward(self, x1m, x5m, sym, stat):
        z = torch.cat([self.branch1(x1m), self.branch5(x5m),
                       self.emb(sym), stat], dim=1)
        h = self.trunk(z)
        return self.head_cls(h), self.head_q(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def score(logits: torch.Tensor) -> torch.Tensor:
        """s = p_bottom(up) - p_top(down): >0 long-at-bottom, <0 short-at-top."""
        p = torch.softmax(logits, dim=1)
        return p[:, 2] - p[:, 0]


def pinball_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over quantiles (0.1, 0.5, 0.9); q is (B,3)."""
    taus = torch.tensor([0.1, 0.5, 0.9], dtype=q.dtype, device=q.device)
    diff = y.unsqueeze(1) - q
    return torch.maximum(taus * diff, (taus - 1) * diff).mean()
