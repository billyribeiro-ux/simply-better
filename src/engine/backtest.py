"""Portfolio simulation over the taken-trade stream.

Position size = (equity x risk%) / stop distance in dollars — constant risk
per trade, geometry-adjusted share count. Slippage is charged against the
trade on both entry and exit; commission per share both sides.

Event-driven accounting (invariant: no lookahead in sizing or risk gates).
A position's realized PnL is credited to equity and to the day's running
loss only when it CLOSES (exit_ts), never when it is entered. Entries are
processed in entry_ts order; before each entry decision, every position
whose exit_ts has already passed is realized. Therefore the equity that
sizes a trade and the day loss that arms the kill switch reflect only
outcomes that were knowable at that entry instant — earlier fix in this
class: the slippage-sign inversion.
"""
from __future__ import annotations

import heapq
import logging
from datetime import date

import numpy as np
import polars as pl

from . import costs
from .config import Config

log = logging.getLogger("engine.backtest")


def run(cfg: Config, taken: pl.DataFrame) -> tuple[pl.DataFrame, list[date], np.ndarray]:
    ex = cfg.execution
    comm = float(ex["commission_per_share"])
    risk_pct = float(ex["risk_per_trade_pct"]) / 100.0
    max_conc = int(ex["max_concurrent"])
    max_daily = float(ex.get("max_daily_loss_pct", 0.0)) / 100.0  # 0 = off
    start_eq = float(ex["starting_equity"])
    equity = start_eq

    if taken.height == 0:
        return pl.DataFrame(), [], np.array([equity])

    df = taken.sort("entry_ts")
    # min-heap of still-open positions, keyed by exit time: (exit_ts, seq, pnl, day)
    open_pos: list = []
    seq = 0
    rows: list[dict] = []
    day_pnl: dict[date, float] = {}
    day_start_eq: dict[date, float] = {}

    def realize_through(ts) -> None:
        """Credit every position that has closed at or before `ts`."""
        nonlocal equity
        while open_pos and open_pos[0][0] <= ts:
            _xexit, _seq, xpnl, xday = heapq.heappop(open_pos)
            equity += xpnl
            day_pnl[xday] = day_pnl.get(xday, 0.0) + xpnl

    for r in df.iter_rows(named=True):
        entry_ts, exit_ts = r["entry_ts"], r["exit_ts"]
        # realize all positions closed by now -> equity/day_pnl are point-in-time
        realize_through(entry_ts)
        if len(open_pos) >= max_conc:
            continue                                    # concurrency cap: skip

        d0 = r["session_date"]
        day_start_eq.setdefault(d0, equity)
        # kill switch: once the day's REALIZED loss (closed trades only)
        # breaches the limit, no new entries open for the rest of the session
        if (max_daily > 0
                and day_pnl.get(d0, 0.0) <= -max_daily * day_start_eq[d0]):
            continue

        side = float(r["side"])
        stop_dist = float(r["stop_atr"]) * float(r["atr"])
        if stop_dist <= 0:
            continue
        risk_usd = equity * risk_pct
        shares = int(risk_usd // stop_dist)
        if shares < 1:
            continue

        # slippage charged AGAINST the trade (invariant 7): longs enter
        # higher / exit lower, shorts enter lower / exit higher. Per-symbol
        # measured spread (costs.slip_frac) — the flat default is a fallback.
        slip = costs.slip_frac(cfg, r.get("symbol", ""))
        entry_fill = float(r["entry_px"]) * (1.0 + side * slip)
        exit_fill = float(r["exit_px"]) * (1.0 - side * slip)
        gross = side * (exit_fill - entry_fill) * shares
        pnl = gross - comm * shares * 2.0

        heapq.heappush(open_pos, (exit_ts, seq, pnl, d0))
        seq += 1
        rows.append({**r, "shares": shares, "pnl_usd": round(pnl, 2),
                     "_exit_key": exit_ts, "_seq": len(rows)})

    # drain any positions still open after the last entry
    realize_through(None if not open_pos else max(p[0] for p in open_pos))

    # equity_after: realized running equity in trade-CLOSE order (honest
    # "equity after this trade settled"); unused downstream but kept truthful
    eq_run = start_eq
    for row in sorted(rows, key=lambda x: (x["_exit_key"], x["_seq"])):
        eq_run += row["pnl_usd"]
        row["equity_after"] = round(eq_run, 2)
    for row in rows:
        del row["_exit_key"], row["_seq"]

    trades = pl.DataFrame(rows) if rows else pl.DataFrame()
    dates = sorted(day_pnl.keys())
    eq = start_eq
    curve = [eq]
    for d in dates:
        eq += day_pnl[d]
        curve.append(eq)
    log.info("backtest: %d trades, final equity %.2f", trades.height, eq)
    return trades, dates, np.array(curve)
