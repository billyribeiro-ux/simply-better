"""Paper trading: resolve live signals to outcomes and keep the forward record.

This is the forward-confirmation ledger. Each session's TAKEN + CONFIRMED
live signals are tracked to their exits with exactly the backtest's rules —
first crossing of stop/target on 1-min bars via ``labeling.scan_paths`` +
``outcome_for`` (intrabar ties against the trade), EOD flat at 15:55,
slippage and commission charged against the trade on both sides. Results
append to an idempotent per-day record (DuckDB ``paper_trades`` + the
dashboard's ``paper.json``).

Trades on a still-open session are marked ``open`` with an unrealized mark
at the last completed close; re-running after the close replaces them with
final outcomes. Only resolved trades enter the KPI record.

The paper record only proves something about signals generated AFTER the
model's ``trained_through`` date on a frozen spec — that is the record the
walk-forward's DSR is waiting for.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timezone
from pathlib import Path

import duckdb
import polars as pl

from . import labeling, live
from .config import Config
from .data import _duck_type
from .production import ProductionBundle

log = logging.getLogger("engine.paper")


def resolve_day(cfg: Config, bundle: ProductionBundle, session_date: date,
                now_et: datetime) -> list[dict]:
    """Resolve the session's taken+confirmed signals to paper trades."""
    report = live.scan_live(cfg, bundle, session_date, now_et)
    taken = [s for s in report["signals"]
             if s["taken"] and s["status"] == "confirmed"]
    if not taken:
        return []

    today, bars1 = live.gather_session(cfg, bundle.universe, session_date, now_et)
    scans = labeling.scan_paths(cfg, today, bars1)
    by_key = {(e["symbol"], e["trigger_ts"].isoformat()): i
              for i, e in enumerate(today)}

    ex = cfg.execution
    slip = float(ex["slippage_bps"]) / 1e4
    comm = float(ex["commission_per_share"])
    grid = cfg.scan_grid
    session_complete = now_et.time() >= time(15, 55)

    rows: list[dict] = []
    for s in taken:
        eid = by_key.get((s["symbol"], s["trigger_ts"]))
        scan = scans.get(eid) if eid is not None else None
        if scan is None:
            continue
        win, pnl_r, reason, xi = labeling.outcome_for(
            scan, grid, float(s["stop_atr"]), float(s["target_atr"]))
        if reason == "eod" and not session_complete:
            reason = "open"          # unrealized mark at the last close
        exit_px = {"target": s["target_px"], "stop": s["stop_px"]}.get(
            reason, float(scan.eod_px))
        exit_ts = scan.cross_ts[xi]

        side = -1.0 if s["side"] == "SHORT" else 1.0
        shares = int(s["shares"])
        entry_fill = float(s["entry_px"]) * (1.0 + side * slip)
        exit_fill = float(exit_px) * (1.0 - side * slip)
        pnl_usd = side * (exit_fill - entry_fill) * shares - comm * shares * 2.0

        rows.append({
            "session_date": session_date.isoformat(),
            "signal_id": s["id"],
            "symbol": s["symbol"], "setup": s["setup"], "side": s["side"],
            "trigger_ts": s["trigger_ts"],
            "entry_ts": s["entry_ts"], "entry_px": float(s["entry_px"]),
            "shares": shares,
            "stop_px": float(s["stop_px"]), "target_px": float(s["target_px"]),
            "exit_ts": exit_ts.isoformat(), "exit_px": round(float(exit_px), 2),
            "exit_reason": reason,
            "resolved": reason != "open",
            "outcome": "WIN" if pnl_usd > 0 else "LOSS",
            "pnl_r": round(float(pnl_r), 3),
            "pnl_usd": round(float(pnl_usd), 2),
            "prob": s["prob"], "threshold": s["threshold"],
            "trained_through": report["trained_through"],
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
    log.info("paper %s: %d taken+confirmed, %d resolved",
             session_date, len(rows), sum(1 for r in rows if r["resolved"]))
    return rows


def persist_day(cfg: Config, session_date: date, rows: list[dict]) -> None:
    """Idempotent per-day append: re-running a day replaces its rows."""
    con = duckdb.connect(str(cfg.db_path))
    try:
        if rows:
            df = pl.DataFrame(rows)
            con.register("paper_df", df.to_arrow())
            con.execute("CREATE TABLE IF NOT EXISTS paper_trades AS "
                        "SELECT * FROM paper_df LIMIT 0")
            existing = {r[1] for r in
                        con.execute("PRAGMA table_info('paper_trades')").fetchall()}
            for name, dtype in zip(df.columns, df.to_arrow().schema.types):
                if name not in existing:
                    con.execute(f'ALTER TABLE paper_trades ADD COLUMN '
                                f'"{name}" {_duck_type(str(dtype))}')
            con.execute("DELETE FROM paper_trades WHERE session_date = ?",
                        [session_date.isoformat()])
            cols = ", ".join(f'"{c}"' for c in df.columns)
            con.execute(f"INSERT INTO paper_trades ({cols}) SELECT * FROM paper_df")
        else:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            if "paper_trades" in tables:
                con.execute("DELETE FROM paper_trades WHERE session_date = ?",
                            [session_date.isoformat()])
    finally:
        con.close()


def export_record(cfg: Config) -> Path:
    """Write the cumulative paper record for the dashboard."""
    con = duckdb.connect(str(cfg.db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        trades = (con.execute(
            "SELECT * FROM paper_trades ORDER BY session_date, entry_ts"
        ).pl() if "paper_trades" in tables else pl.DataFrame())
    finally:
        con.close()

    equity0 = float(cfg.execution["starting_equity"])
    resolved = (trades.filter(pl.col("resolved")) if trades.height
                else pl.DataFrame())
    kpis: dict = {"trades": int(resolved.height), "open": 0,
                  "win_rate": None, "net_pnl_usd": 0.0, "expectancy_r": None}
    curve_dates: list[str] = []
    curve = [equity0]
    if trades.height:
        kpis["open"] = int(trades.filter(~pl.col("resolved")).height)
    if resolved.height:
        pnl = resolved["pnl_usd"].to_numpy()
        kpis["win_rate"] = round(float((pnl > 0).mean()) * 100.0, 2)
        kpis["net_pnl_usd"] = round(float(pnl.sum()), 2)
        kpis["expectancy_r"] = round(float(resolved["pnl_r"].mean()), 4)
        eq = equity0
        for (d,), g in resolved.group_by(["session_date"], maintain_order=True):
            eq += float(g["pnl_usd"].sum())
            curve_dates.append(str(d))
            curve.append(round(eq, 2))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity_start": equity0,
        "kpis": kpis,
        "equity": {"dates": curve_dates, "curve": curve},
        "trades": trades.to_dicts() if trades.height else [],
    }
    path = cfg.root / cfg.raw.get("live", {}).get(
        "paper_file", "dashboard/static/data/paper.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return path
