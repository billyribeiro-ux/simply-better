"""1-min entry refinement and path scanning.

Two-stage labeling, which is what makes the geometry learnable instead of
assumed:

1. ``scan_paths``  — enter on 1-min break confirmation, then record, with NO
   barriers attached, the first 1-min bar index at which adverse / favorable
   excursion crosses every level of a fixed ATR-multiple grid. Those two
   integer arrays per trade contain the complete outcome of *any* stop/target
   pair on the grid — so geometry search later is pure array math, no
   re-simulation, no lookahead.
2. ``label``       — given a chosen (stop, target) pair (grid values), derive
   win/loss/EOD outcome, R-multiple pnl, MAE/MFE, exit time & reason.

Intrabar ambiguity (stop and target crossed on the same 1-min bar) is always
resolved AGAINST the trade. Conservative by construction.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from .config import Config

log = logging.getLogger("engine.labeling")


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


class PathScan:
    """Per-event first-crossing record over the ATR-multiple grid."""

    __slots__ = ("event_id", "entry_ts", "entry_px", "adv_cross", "fav_cross",
                 "eod_signed_atr", "eod_ts", "eod_px", "cross_ts",
                 "fav_path", "adv_path", "close_path")

    def __init__(self, event_id: int, entry_ts: datetime, entry_px: float,
                 adv_cross: np.ndarray, fav_cross: np.ndarray,
                 eod_signed_atr: float, eod_ts: datetime, eod_px: float,
                 cross_ts: list[datetime],
                 fav_path: np.ndarray | None = None,
                 adv_path: np.ndarray | None = None,
                 close_path: np.ndarray | None = None) -> None:
        self.event_id = event_id
        self.entry_ts = entry_ts
        self.entry_px = entry_px
        self.adv_cross = adv_cross      # int index into 1-min path, -1 = never
        self.fav_cross = fav_cross
        self.eod_signed_atr = eod_signed_atr
        self.eod_ts = eod_ts
        self.eod_px = eod_px
        self.cross_ts = cross_ts        # bar timestamps of the walked path
        # research instrumentation (float32, per-bar, entry -> flat_by):
        # signed excursions in ATR units and raw closes. Exit research MUST
        # stay zero-lookahead: decisions at bar t use paths[:t+1] only, and
        # same-bar ties resolve against the trade (mirror outcome_for).
        self.fav_path = fav_path
        self.adv_path = adv_path
        self.close_path = close_path


def scan_paths(cfg: Config, events: list[dict],
               bars1_by_symbol: dict[str, pl.DataFrame]) -> dict[int, PathScan]:
    """Refine entries on 1-min and scan barrier-free excursion paths."""
    grid = cfg.scan_grid
    confirm_n = int(cfg.entry["confirm_bars_1m"])
    tick = float(cfg.entry["tick"])
    flat_by = _hm(cfg.session["flat_by"])

    # pre-split each symbol's 1-min bars by session date for O(1) lookup
    day_cache: dict[tuple[str, object], dict[str, np.ndarray]] = {}
    for sym, df in bars1_by_symbol.items():
        if df.height == 0:
            continue
        for (day,), g in (
            df.with_columns(pl.col("ts").dt.date().alias("d"))
            .sort("ts").group_by(["d"], maintain_order=True)
        ):
            day_cache[(sym, day)] = {
                "ts": np.array(g["ts"].to_list(), dtype=object),
                "o": g["open"].to_numpy().astype(float),
                "h": g["high"].to_numpy().astype(float),
                "l": g["low"].to_numpy().astype(float),
                "c": g["close"].to_numpy().astype(float),
            }

    scans: dict[int, PathScan] = {}
    for eid, ev in enumerate(events):
        key = (ev["symbol"], ev["session_date"])
        d = day_cache.get(key)
        if d is None:
            continue
        ts, o, h, lo, c = d["ts"], d["o"], d["h"], d["l"], d["c"]

        # first 1-min bar strictly after the 5-min trigger bar closes
        trig_end = ev["trigger_ts"] + timedelta(minutes=5)
        start = int(np.searchsorted(
            np.array([t.hour * 60 + t.minute for t in ts]),
            trig_end.hour * 60 + trig_end.minute, side="left"))
        if start >= len(ts):
            continue

        side = float(ev["side"])
        atr = float(ev["atr"])

        # ---- break confirmation ------------------------------------------
        entry_idx, entry_px = -1, np.nan
        if side < 0:
            brk = float(ev["trigger_low"]) - tick
            for j in range(start, min(start + confirm_n, len(ts))):
                if lo[j] <= brk:
                    entry_idx = j
                    entry_px = min(o[j], brk)   # stop-market fill
                    break
        else:
            brk = float(ev["trigger_high"]) + tick
            for j in range(start, min(start + confirm_n, len(ts))):
                if h[j] >= brk:
                    entry_idx = j
                    entry_px = max(o[j], brk)
                    break
        if entry_idx < 0:
            continue    # never confirmed — no trade, event dropped

        # ---- walk to flat_by ----------------------------------------------
        mins = np.array([t.hour * 60 + t.minute for t in ts[entry_idx:]])
        stop_at = int(np.searchsorted(mins, flat_by, side="left"))
        if stop_at < 1:
            continue
        seg = slice(entry_idx, entry_idx + stop_at)
        hh, ll, cc = h[seg], lo[seg], c[seg]
        path_ts = list(ts[seg])

        if side < 0:
            fav = (entry_px - ll) / atr
            adv = (hh - entry_px) / atr
            eod_signed = (entry_px - cc[-1]) / atr
        else:
            fav = (hh - entry_px) / atr
            adv = (entry_px - ll) / atr
            eod_signed = (cc[-1] - entry_px) / atr

        run_fav = np.maximum.accumulate(fav)
        run_adv = np.maximum.accumulate(adv)
        # run_* are non-decreasing -> searchsorted gives first crossing index
        fav_cross = np.searchsorted(run_fav, grid, side="left")
        adv_cross = np.searchsorted(run_adv, grid, side="left")
        L = len(run_fav)
        fav_cross = np.where(fav_cross >= L, -1, fav_cross).astype(np.int32)
        adv_cross = np.where(adv_cross >= L, -1, adv_cross).astype(np.int32)

        scans[eid] = PathScan(
            event_id=eid, entry_ts=ts[entry_idx], entry_px=float(entry_px),
            adv_cross=adv_cross, fav_cross=fav_cross,
            eod_signed_atr=float(eod_signed),
            eod_ts=path_ts[-1], eod_px=float(cc[-1]), cross_ts=path_ts,
            fav_path=fav.astype(np.float32), adv_path=adv.astype(np.float32),
            close_path=cc.astype(np.float32),
        )
    log.info("confirmed entries: %d / %d events", len(scans), len(events))
    return scans


def outcome_for(scan: PathScan, grid: np.ndarray,
                stop_atr: float, target_atr: float) -> tuple[int, float, str, int]:
    """(win, pnl_r, exit_reason, exit_path_idx) for one geometry choice."""
    si = int(np.argmin(np.abs(grid - stop_atr)))
    ti = int(np.argmin(np.abs(grid - target_atr)))
    a, f = int(scan.adv_cross[si]), int(scan.fav_cross[ti])
    if f != -1 and (a == -1 or f < a):          # ties resolve to the stop
        return 1, target_atr / stop_atr, "target", f
    if a != -1:
        return 0, -1.0, "stop", a
    return (1 if scan.eod_signed_atr > 0 else 0,
            scan.eod_signed_atr / stop_atr, "eod", len(scan.cross_ts) - 1)


def label(cfg: Config, events: list[dict], scans: dict[int, PathScan],
          geometry_lookup) -> pl.DataFrame:
    """Apply learned geometry to every confirmed event -> labeled trades."""
    grid = cfg.scan_grid
    rows: list[dict] = []
    for eid, ev in enumerate(events):
        scan = scans.get(eid)
        if scan is None:
            continue
        stop_atr, target_atr, tradable = geometry_lookup(ev)
        if not tradable:
            continue    # anchor geometry refused it — same status as unconfirmed
        win, pnl_r, reason, xi = outcome_for(scan, grid, stop_atr, target_atr)
        atr, side = float(ev["atr"]), float(ev["side"])
        stop_px = scan.entry_px + side * -1 * stop_atr * atr
        target_px = scan.entry_px + side * target_atr * atr
        if reason == "target":
            exit_px = target_px
        elif reason == "stop":
            exit_px = stop_px
        else:
            exit_px = scan.eod_px
        # realized MAE/MFE up to exit, in R units
        si = int(np.argmin(np.abs(grid - stop_atr)))
        upto = xi + 1
        mae_r = _max_excursion(scan.adv_cross, grid, upto) / stop_atr
        mfe_r = _max_excursion(scan.fav_cross, grid, upto) / stop_atr
        rows.append({
            "event_id": eid, "symbol": ev["symbol"],
            "session_date": ev["session_date"], "setup": ev["setup"],
            "side": side, "trigger_ts": ev["trigger_ts"],
            "entry_ts": scan.entry_ts, "entry_px": scan.entry_px,
            "stop_atr": stop_atr, "target_atr": target_atr,
            "stop_px": float(stop_px), "target_px": float(target_px),
            "exit_ts": scan.cross_ts[xi], "exit_px": float(exit_px),
            "exit_reason": reason, "outcome": int(win),
            "pnl_r": float(pnl_r), "mae_r": float(mae_r), "mfe_r": float(mfe_r),
            "atr": atr,
            # decision-layer gate input — deliberately NOT in FEATURES
            "dt_eff_h1_aligned": float(ev.get("dt_eff_h1_aligned", 0.0)),
            **{k: float(ev[k]) for k in _FEATURE_KEYS},
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _max_excursion(cross: np.ndarray, grid: np.ndarray, upto: int) -> float:
    """Largest grid level whose first crossing happened before bar `upto`."""
    hit = (cross != -1) & (cross < upto)
    return float(grid[hit].max()) if hit.any() else 0.0


_FEATURE_KEYS = [
    "minutes_since_open", "tod_sin", "tod_cos", "gap_pct", "range_ext_atr",
    "vol_z", "dist_vwap_atr", "prior_high_dist_atr", "prior_low_dist_atr",
    "gk_vol_z", "ret_30m_atr", "wick_ratio", "n_extremes", "dow", "atr_pct",
    "room_vwap_atr", "room_open_atr", "room_pclose_atr", "room_pdpoc_atr",
    "room_dpoc_atr", "vwap_slope_atr", "max_room_atr",
]
