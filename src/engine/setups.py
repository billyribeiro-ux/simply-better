"""Primary-signal layer: HOD-fade and LOD-reclaim detection on 5-min bars.

Every emitted event carries its full feature vector computed strictly from
information available at the trigger bar's close. The ML meta-model then
decides which triggers are worth trading (meta-labeling architecture).
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import polars as pl

from .anchors import PriceHistogram, anchor_prices, anchor_room_atr
from .config import Config

log = logging.getLogger("engine.setups")

HOD_FADE = "HOD_FADE"
LOD_RECLAIM = "LOD_RECLAIM"


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def detect(cfg: Config, symbol: str, bars5: pl.DataFrame,
           ctx: pl.DataFrame) -> list[dict]:
    """Scan one symbol's 5-min history and emit setup events with features."""
    if bars5.height == 0 or ctx.height == 0:
        return []

    scfg = cfg.setups
    hod_cfg, lod_cfg = scfg["hod_fade"], scfg["lod_reclaim"]
    cooldown = int(scfg["cooldown_bars"])
    max_side = int(scfg["max_per_side_per_day"])
    min_trig = _hm(cfg.session["min_trigger"])
    no_new = _hm(cfg.session["no_new_after"])

    ctx_map: dict = {r["date"]: r for r in ctx.iter_rows(named=True)}

    df = bars5.with_columns(pl.col("ts").dt.date().alias("d")).sort("ts")
    events: list[dict] = []

    # prior-day volume-at-price histogram, carried across the day loop
    pd_hist: PriceHistogram | None = None

    for (day,), g in df.group_by(["d"], maintain_order=True):
        c = ctx_map.get(day)
        if c is None or c["atr"] is None or not np.isfinite(c["atr"]) or c["atr"] <= 0:
            continue
        atr = float(c["atr"])
        prior_close = float(c["prior_close"]) if c["prior_close"] else np.nan
        prior_high = float(c["prior_high"]) if c["prior_high"] else np.nan
        prior_low = float(c["prior_low"]) if c["prior_low"] else np.nan
        gk_z = float(c["gk_vol_z"]) if c["gk_vol_z"] is not None else 0.0

        ts = g["ts"].to_list()
        o = g["open"].to_numpy().astype(float)
        h = g["high"].to_numpy().astype(float)
        lo = g["low"].to_numpy().astype(float)
        cl = g["close"].to_numpy().astype(float)
        v = g["volume"].to_numpy().astype(float)
        n = len(ts)
        if n < 8:
            continue

        day_open = o[0]
        gap_pct = (day_open / prior_close - 1.0) * 100.0 if np.isfinite(prior_close) else 0.0

        hod, lod = h[0], lo[0]
        n_new_highs, n_new_lows = 0, 0
        cum_pv, cum_v = 0.0, 0.0
        last_trig = {HOD_FADE: -10**9, LOD_RECLAIM: -10**9}
        count = {HOD_FADE: 0, LOD_RECLAIM: 0}

        # anchors: prior-day POC from the carried histogram, developing POC
        # built incrementally (bin width fixed at session start from prior ATR)
        pd_poc = pd_hist.poc() if pd_hist is not None and not pd_hist.is_empty() else None
        day_hist = PriceHistogram(atr)
        vwap_hist: list[float] = []

        # day-type: directional efficiency of the session, frozen by WALL
        # CLOCK at the last 5-min bar closing <= 10:30 (bar-index freezing
        # breaks on sessions with non-contiguous first hours); before the
        # freeze it is efficiency-so-far
        cum_path = 0.0
        eff_frozen = 0.0

        for i in range(n):
            # dev POC at bar i uses volume from bars 0..i-1 ONLY, then bar i
            # is added — the sequencing that keeps it lookahead-free
            dev_poc = day_hist.poc()
            day_hist.add_bar(h[i], lo[i], cl[i], v[i])

            typ = (h[i] + lo[i] + cl[i]) / 3.0
            cum_pv += typ * v[i]
            cum_v += v[i]
            vwap = cum_pv / cum_v if cum_v > 0 else cl[i]
            vwap_hist.append(vwap)

            cum_path += abs(cl[i] - day_open) if i == 0 else abs(cl[i] - cl[i - 1])
            eff_i = (cl[i] - day_open) / cum_path if cum_path > 0 else 0.0
            bar_close_mins = ts[i].hour * 60 + ts[i].minute + 5
            if bar_close_mins <= 10 * 60 + 30:
                eff_frozen = eff_i

            new_hod = h[i] > hod
            new_lod = lo[i] < lod
            if new_hod:
                hod = h[i]
                n_new_highs += 1
            if new_lod:
                lod = lo[i]
                n_new_lows += 1
            if i == 0:
                continue

            t: datetime = ts[i]
            mins = t.hour * 60 + t.minute
            if mins < min_trig or mins >= no_new:
                continue

            rng = h[i] - lo[i]
            if rng <= 0:
                continue
            upper_wick = h[i] - max(o[i], cl[i])
            lower_wick = min(o[i], cl[i]) - lo[i]

            # volume z vs day-so-far (needs a few bars of history)
            if i >= 5:
                vm, vs = float(np.mean(v[:i])), float(np.std(v[:i], ddof=1))
                vol_z = (v[i] - vm) / vs if vs > 0 else 0.0
            else:
                vol_z = 0.0

            ret_30m = (cl[i] - cl[max(0, i - 6)]) / atr
            minutes_since_open = mins - _hm(cfg.session["open"])
            frac = minutes_since_open / (6.5 * 60.0)

            apx = anchor_prices(
                vwap, float(day_open),
                float(prior_close) if np.isfinite(prior_close) else None,
                pd_poc, dev_poc,
            )
            vwap_slope = (vwap - vwap_hist[i - 6]) / atr if i >= 6 else 0.0

            def rooms_for(side: float) -> dict[str, float]:
                return {
                    "room_vwap_atr": anchor_room_atr(apx["vwap"], cl[i], side, atr),
                    "room_open_atr": anchor_room_atr(apx["day_open"], cl[i], side, atr),
                    "room_pclose_atr": anchor_room_atr(apx["prior_close"], cl[i], side, atr),
                    "room_pdpoc_atr": anchor_room_atr(apx["pd_poc"], cl[i], side, atr),
                    "room_dpoc_atr": anchor_room_atr(apx["dev_poc"], cl[i], side, atr),
                }

            base = {
                "symbol": symbol,
                "session_date": day,
                "trigger_ts": t,
                "trigger_close": cl[i],
                "trigger_high": h[i],
                "trigger_low": lo[i],
                "atr": atr,
                "gk_vol_z": gk_z,
                "minutes_since_open": float(minutes_since_open),
                "tod_sin": float(np.sin(2 * np.pi * frac)),
                "tod_cos": float(np.cos(2 * np.pi * frac)),
                "gap_pct": float(gap_pct),
                "vol_z": float(vol_z),
                "dist_vwap_atr": float((cl[i] - vwap) / atr),
                "prior_high_dist_atr": float((prior_high - cl[i]) / atr)
                if np.isfinite(prior_high) else 0.0,
                "prior_low_dist_atr": float((cl[i] - prior_low) / atr)
                if np.isfinite(prior_low) else 0.0,
                "ret_30m_atr": float(ret_30m),
                "dow": float(day.weekday()),
                "atr_pct": float(atr / prior_close * 100.0)
                if np.isfinite(prior_close) and prior_close > 0 else 0.0,
                "anchor_px": apx,
                "vwap_slope_atr": float(vwap_slope),
            }

            # ---- HOD FADE (short) ----------------------------------------
            if (
                new_hod
                and cl[i] < o[i]
                and (h[i] - day_open) / atr >= float(hod_cfg["ext_min_atr"])
                and upper_wick / rng >= float(hod_cfg["wick_frac_min"])
                and count[HOD_FADE] < max_side
                and i - last_trig[HOD_FADE] >= cooldown
            ):
                rooms = rooms_for(-1.0)
                events.append({
                    **base,
                    **rooms,
                    "max_room_atr": float(max(rooms.values())),
                    # aligned first-hour efficiency: positive = session trending
                    # against this fade (NOT in FEATURES — decision-layer gate)
                    "dt_eff_h1_aligned": float(-(-1.0) * eff_frozen),
                    "setup": HOD_FADE,
                    "side": -1.0,
                    "level": float(hod),
                    "range_ext_atr": float((h[i] - day_open) / atr),
                    "wick_ratio": float(upper_wick / rng),
                    "n_extremes": float(n_new_highs),
                })
                last_trig[HOD_FADE] = i
                count[HOD_FADE] += 1

            # ---- LOD RECLAIM (long) --------------------------------------
            if (
                new_lod
                and cl[i] > o[i]
                and (day_open - lo[i]) / atr >= float(lod_cfg["ext_min_atr"])
                and lower_wick / rng >= float(lod_cfg["wick_frac_min"])
                and count[LOD_RECLAIM] < max_side
                and i - last_trig[LOD_RECLAIM] >= cooldown
            ):
                rooms = rooms_for(1.0)
                events.append({
                    **base,
                    **rooms,
                    "max_room_atr": float(max(rooms.values())),
                    "dt_eff_h1_aligned": float(-(1.0) * eff_frozen),
                    "setup": LOD_RECLAIM,
                    "side": 1.0,
                    "level": float(lod),
                    "range_ext_atr": float((day_open - lo[i]) / atr),
                    "wick_ratio": float(lower_wick / rng),
                    "n_extremes": float(n_new_lows),
                })
                last_trig[LOD_RECLAIM] = i
                count[LOD_RECLAIM] += 1

        # finished session becomes the prior-day histogram for the next one
        pd_hist = day_hist

    log.info("[%s] %d setup events", symbol, len(events))
    return events
