"""Portfolio simulation over the taken-trade stream.

Position size = (equity x risk%) / stop distance in dollars — constant risk
per trade, geometry-adjusted share count. Slippage is charged against the
trade on both entry and exit; commission per share both sides.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

from .config import Config

log = logging.getLogger("engine.backtest")


def run(cfg: Config, taken: pl.DataFrame) -> tuple[pl.DataFrame, list[date], np.ndarray]:
    ex = cfg.execution
    slip = float(ex["slippage_bps"]) / 1e4
    comm = float(ex["commission_per_share"])
    risk_pct = float(ex["risk_per_trade_pct"]) / 100.0
    max_conc = int(ex["max_concurrent"])
    equity = float(ex["starting_equity"])

    if taken.height == 0:
        return pl.DataFrame(), [], np.array([equity])

    df = taken.sort("entry_ts")
    open_exits: list = []          # exit timestamps of live positions
    rows: list[dict] = []
    day_pnl: dict[date, float] = {}

    for r in df.iter_rows(named=True):
        entry_ts, exit_ts = r["entry_ts"], r["exit_ts"]
        open_exits = [x for x in open_exits if x > entry_ts]
        if len(open_exits) >= max_conc:
            continue                                    # concurrency cap: skip

        side = float(r["side"])
        stop_dist = float(r["stop_atr"]) * float(r["atr"])
        if stop_dist <= 0:
            continue
        risk_usd = equity * risk_pct
        shares = int(risk_usd // stop_dist)
        if shares < 1:
            continue

        entry_fill = float(r["entry_px"]) * (1.0 - side * slip)   # against us
        exit_fill = float(r["exit_px"]) * (1.0 + side * slip)     # against us
        gross = side * (exit_fill - entry_fill) * shares
        pnl = gross - comm * shares * 2.0
        equity += pnl

        d = r["session_date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + pnl
        open_exits.append(exit_ts)

        rows.append({**r, "shares": shares, "pnl_usd": round(pnl, 2),
                     "equity_after": round(equity, 2)})

    trades = pl.DataFrame(rows) if rows else pl.DataFrame()
    dates = sorted(day_pnl.keys())
    eq = float(ex["starting_equity"])
    curve = [eq]
    for d in dates:
        eq += day_pnl[d]
        curve.append(eq)
    log.info("backtest: %d trades, final equity %.2f", trades.height, eq)
    return trades, dates, np.array(curve)
