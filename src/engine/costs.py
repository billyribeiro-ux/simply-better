"""Execution-cost model — the single source of truth for per-symbol slippage
and the net-of-cost expected value used to gate trades.

Shared verbatim by `backtest.run`, the research decide loop (`learn`), and the
live scanner (`live`) so cost handling can never skew between simulate and
serve. The root-cause audit (2026-07-08 ledger) established that the strategy's
gross edge is real but thin and sits on the transaction-cost boundary; charging
each name its MEASURED effective spread (not a flat retail default) and taking
only trades whose expected value clears that cost is the disciplined response.

`slippage_bps_by_symbol` is a measured input (Corwin-Schultz 2012 high-low
effective half-spread on the 5-min bars, a conservative upper bound) — it is
NOT tuned to P&L, and lowering it to manufacture profit is banned as tuning.
"""
from __future__ import annotations

from .config import Config


def slip_frac(cfg: Config, symbol: str) -> float:
    """Per-symbol slippage as a fraction of price (bps / 1e4). Falls back to the
    flat ``execution.slippage_bps`` for any symbol without a measured entry."""
    ex = cfg.execution
    by = ex.get("slippage_bps_by_symbol") or {}
    bps = float(by.get(symbol, ex["slippage_bps"]))
    return bps / 1e4


def cost_r(slip: float, comm: float, px: float,
           stop_atr: float, atr: float) -> float:
    """Modeled round-trip execution cost expressed in R.

    cost_R = 2 * (slip * px + comm) / (stop_atr * atr). Share count cancels, so
    the cost falls as the stop widens. Uses entry price, ATR, and config
    constants only — no lookahead.
    """
    denom = stop_atr * atr
    if denom <= 0:
        return 0.0
    return 2.0 * (slip * px + comm) / denom


def net_ev_r(p: float, stop_atr: float, target_atr: float, cost: float) -> float:
    """Expected value per trade in R, net of cost, under the win/loss binary the
    geometry targets: ``p * (target/stop) - (1 - p) - cost``. A trade clears the
    net-EV gate when this is strictly positive."""
    if stop_atr <= 0:
        return -1e9
    return p * (target_atr / stop_atr) - (1.0 - p) - cost
