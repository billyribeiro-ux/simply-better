"""Live signal generation — the production twin of the research loop.

Every stage reuses the validated research code verbatim: ``setups.detect``
for triggers and features, ``model.score`` for probabilities,
``Geometry.lookup`` for stop/target/tradability, ``labeling.scan_paths`` for
1-min break confirmation, and the backtester's sizing arithmetic. The only
live-specific logic is data freshness (drop the in-progress bar) and signal
status bookkeeping. Train/serve parity is asserted by tests/test_live_path.py.

Attribution rules are applied with a sentinel fold ("9999-12") that is
strictly later than any learnable fold string: at live time every rule was
mined from data that is entirely in the past, so all of them apply — the
same relationship every historical fold had to the rules before it.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import polars as pl

from . import costs, labeling, model, setups
from .attribution import Attribution
from .config import Config
from .data import FMPClient, Store, _duck_type
from .features import FEATURES, daily_context
from .production import ProductionBundle

log = logging.getLogger("engine.live")

ET = ZoneInfo("America/New_York")
LIVE_FOLD = "9999-12"           # strictly later than every learnable fold
_LOOKBACK_CAL_DAYS = 14         # calendar window covering lookback sessions


def now_et_naive() -> datetime:
    """Current US/Eastern wall-clock, naive (the engine's timestamp basis)."""
    return datetime.now(ET).replace(tzinfo=None)


def refresh_data(cfg: Config, session_date: date) -> None:
    """Pull the freshest daily/5-min/1-min bars needed to scan one session."""
    client = FMPClient(cfg.fmp_api_key)
    store = Store(cfg)
    daily_from = session_date - timedelta(days=140)
    intraday_from = session_date - timedelta(days=_LOOKBACK_CAL_DAYS)
    try:
        for sym in cfg.universe:
            store.save_daily(sym, client.daily(sym, daily_from, session_date))
            store.save_bars("5min", sym, client.intraday(
                sym, "5min", intraday_from, session_date))
            store.save_bars("1min", sym, client.intraday(
                sym, "1min", session_date, session_date))
    finally:
        client.close()


def gather_session(cfg: Config, universe: list[str], session_date: date,
                   now_et: datetime) -> tuple[list[dict], dict[str, pl.DataFrame]]:
    """Detect a session's trigger events and load its completed 1-min bars.

    Shared by scan_live and the paper resolver so decisions and outcome
    resolution always see byte-identical data."""
    store = Store(cfg)
    lookback_start = session_date - timedelta(days=_LOOKBACK_CAL_DAYS)
    events: list[dict] = []
    bars1: dict[str, pl.DataFrame] = {}
    for sym in universe:
        ctx = daily_context(store.load_daily(sym))
        b5 = store.load_bars("5min", sym, lookback_start, session_date)
        # live hygiene: only completed bars may be seen
        b5 = b5.filter(pl.col("ts") + pl.duration(minutes=5) <= now_et)
        events.extend(setups.detect(cfg, sym, b5, ctx))
        b1 = store.load_bars("1min", sym, session_date, session_date)
        bars1[sym] = b1.filter(pl.col("ts") + pl.duration(minutes=1) <= now_et)
    today = sorted((e for e in events if e["session_date"] == session_date),
                   key=lambda e: e["trigger_ts"])
    return today, bars1


def scan_live(cfg: Config, bundle: ProductionBundle, session_date: date,
              now_et: datetime) -> dict:
    """Detect, score, and size today's triggers with the production bundle."""
    tick = float(cfg.entry["tick"])
    confirm_n = int(cfg.entry["confirm_bars_1m"])
    ex = cfg.execution
    equity = float(ex["starting_equity"])
    risk_usd = equity * float(ex["risk_per_trade_pct"]) / 100.0
    comm = float(ex["commission_per_share"])
    fixed_shares = int(ex.get("fixed_shares", 0))   # 0 = risk-based sizing
    net_gate = bool(cfg.setups.get("net_ev_gate", False))

    today, bars1 = gather_session(cfg, bundle.universe, session_date, now_et)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_date": session_date.isoformat(),
        "as_of_et": now_et.isoformat(),
        "trained_through": bundle.trained_through.isoformat(),
        "threshold_base": round(bundle.fm.threshold, 3),
        "equity": equity,
        "signals": [],
    }
    if not any(bars1[s].height or 0 for s in bundle.universe) and not today:
        report["note"] = (f"no bars for {session_date} — market closed, "
                          "data not yet available, or outside session")
        return report

    if not today:
        report["note"] = "no setup triggers so far this session"
        return report

    X = np.array([[float(e[f]) for f in FEATURES] for e in today])
    probs = model.score(bundle.fm, X)
    scans = labeling.scan_paths(cfg, today, bars1)
    attribution = Attribution(rules=list(bundle.rules))
    gate_max = float(cfg.setups.get("day_type", {}).get("eff_gate_max", 1e9))

    signals: list[dict] = []
    for j, ev in enumerate(today):
        side = float(ev["side"])
        atr = float(ev["atr"])
        stop_atr, target_atr, tradable = bundle.geometry.lookup(ev)
        cell = bundle.geometry.cells.get(
            (ev["setup"], bundle.geometry.regime_of(float(ev["gk_vol_z"]))),
            bundle.geometry.fallback)
        eff_thr = attribution.effective_threshold(ev, bundle.fm.threshold, LIVE_FOLD)
        prob = float(probs[j])
        # same trend veto as the research decide loop (frozen config)
        trend_veto = float(ev.get("dt_eff_h1_aligned", 0.0)) > gate_max
        taken = bool(tradable and prob >= eff_thr and not trend_veto)

        # planned entry = the 1-min break level (stop-market order price)
        brk = (float(ev["trigger_low"]) - tick if side < 0
               else float(ev["trigger_high"]) + tick)
        # net-of-cost EV gate, identical to the research decide loop. Uses the
        # confirmed fill when available else the planned break level.
        ev_px = float(scans[j].entry_px) if scans.get(j) is not None else brk
        net_ev = costs.net_ev_r(
            prob, float(stop_atr), float(target_atr),
            costs.cost_r(costs.slip_frac(cfg, ev["symbol"]), comm, ev_px,
                         float(stop_atr), float(atr)))
        if net_gate:
            taken = taken and net_ev > 0.0
        scan = scans.get(j)
        if scan is not None:
            status, entry_ts, entry_px = "confirmed", scan.entry_ts, scan.entry_px
        else:
            trig_end = ev["trigger_ts"] + timedelta(minutes=5)
            sym_bars = bars1.get(ev["symbol"])
            elapsed = (sym_bars.filter(pl.col("ts") >= trig_end).height
                       if sym_bars is not None and sym_bars.height else 0)
            status = "expired" if elapsed >= confirm_n else "awaiting"
            entry_ts, entry_px = None, None

        entry_ref = float(entry_px) if entry_px is not None else brk
        stop_dist = stop_atr * atr
        stop_px = entry_ref + side * -1 * stop_atr * atr
        target_px = entry_ref + side * target_atr * atr
        if fixed_shares > 0:
            shares = fixed_shares          # owner order: flat 1-share trades
        else:
            shares = int(risk_usd // stop_dist) if stop_dist > 0 else 0

        signals.append({
            "id": f"{ev['symbol']}-{ev['trigger_ts']:%H%M}",
            "symbol": ev["symbol"],
            "setup": ev["setup"],
            "side": "SHORT" if side < 0 else "LONG",
            "trigger_ts": ev["trigger_ts"].isoformat(),
            "prob": round(prob, 3),
            "threshold": round(float(eff_thr), 3),
            "taken": taken,
            "tradable": bool(tradable),
            "trend_veto": bool(trend_veto),
            "day_type_eff": round(float(ev.get("dt_eff_h1_aligned", 0.0)), 3),
            "anchor": cell.anchor,
            "frac": cell.frac,
            "stop_atr": round(float(stop_atr), 2),
            "target_atr": round(float(target_atr), 2),
            "entry_trigger_px": round(brk, 2),
            "stop_px": round(float(stop_px), 2),
            "target_px": round(float(target_px), 2),
            "shares": shares,
            "net_ev_r": round(float(net_ev), 4),
            "status": status,
            "entry_ts": entry_ts.isoformat() if entry_ts is not None else None,
            "entry_px": round(float(entry_px), 2) if entry_px is not None else None,
            "max_room_atr": round(float(ev["max_room_atr"]), 2),
        })

    # MIE-DL stream (setup DL_SEQ) — only when the pre-registered gates have
    # flipped dl.enabled; identical schema, appended to the same feed
    if bool(cfg.raw.get("dl", {}).get("enabled", False)):
        try:
            from .dl.signals import scan_live_dl
            signals.extend(scan_live_dl(cfg, session_date, now_et))
        except ImportError as exc:
            log.warning("dl.enabled but torch missing: %s", exc)

    report["signals"] = signals
    log.info("live scan %s as of %s: %d triggers, %d taken",
             session_date, now_et.time(), len(signals),
             sum(1 for s in signals if s["taken"]))
    return report


def write_live_json(cfg: Config, report: dict) -> Path:
    out = cfg.root / cfg.raw.get("live", {}).get(
        "out_file", "dashboard/static/data/live.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, separators=(",", ":"))
    return out


def persist_live(cfg: Config, report: dict) -> None:
    """Append-only live-signal record (same widening rule as the trades table)."""
    if not report["signals"]:
        return
    df = pl.DataFrame(report["signals"]).with_columns(
        pl.lit(report["generated_at"]).alias("emitted_at"),
        pl.lit(report["session_date"]).alias("session_date"),
    )
    con = duckdb.connect(str(cfg.db_path))
    try:
        con.register("live_df", df.to_arrow())
        con.execute("CREATE TABLE IF NOT EXISTS live_signals AS "
                    "SELECT * FROM live_df LIMIT 0")
        existing = {r[1] for r in
                    con.execute("PRAGMA table_info('live_signals')").fetchall()}
        for name, dtype in zip(df.columns, df.to_arrow().schema.types):
            if name not in existing:
                con.execute(f'ALTER TABLE live_signals ADD COLUMN '
                            f'"{name}" {_duck_type(str(dtype))}')
        cols = ", ".join(f'"{c}"' for c in df.columns)
        con.execute(f"INSERT INTO live_signals ({cols}) SELECT * FROM live_df")
    finally:
        con.close()
