"""Score -> trade construction (frozen a priori, ledger 2026-07-09).

s = p_bottom - p_top. Trade when |s| >= s_min. Entry = next 1-min open after
the decision bar. Stop = 1.0 vol-unit, target = 1.0 vol-unit (u = sigma_t *
sqrt(H) * close_t dollars/share), 30-min time exit, unconditional flat by
15:55, intrabar stop/target ties resolve AGAINST the trade (mirrors
labeling.outcome_for). Net-EV gate reuses engine/costs.py verbatim. The same
functions drive evaluation replay and live scanning — parity by construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from .dataset import Samples, SymbolCache, SYMBOLS_ORDER
from .. import costs
from ..config import Config

log = logging.getLogger("engine.dl.signals")

FLAT_BY_MIN = 955          # 15:55 — bars stamped < 15:55 close by 15:55


@dataclass
class PathOutcome:
    entry_idx: int
    entry_px: float
    exit_idx: int
    exit_px: float
    exit_reason: str        # target | stop | time | eod
    win: int                # target-before-stop (time/eod exits: sign of pnl)
    pnl_r_gross: float      # R multiple before costs (stop distance = 1R)


def walk_path(cache: SymbolCache, i_dec: int, side: float, unit: float,
              k_stop: float, k_target: float, time_exit_min: int
              ) -> PathOutcome | None:
    """Walk 1-min bars after the decision bar. Returns None when no entry bar
    exists in-session (decision too close to the bell — excluded upstream)."""
    j0 = i_dec + 1
    n = len(cache.ts1)
    if j0 >= n or cache.date1[j0] != cache.date1[i_dec]:
        return None
    entry = float(cache.open1[j0])
    stop_px = entry - side * k_stop * unit
    target_px = entry + side * k_target * unit

    mins0 = int((cache.ts1[j0].astype("datetime64[m]")
                 - cache.ts1[j0].astype("datetime64[D]")).astype(int))
    j = j0
    last = j0
    timed_out = False
    while j < n and cache.date1[j] == cache.date1[i_dec]:
        m = int((cache.ts1[j].astype("datetime64[m]")
                 - cache.ts1[j].astype("datetime64[D]")).astype(int))
        if m >= FLAT_BY_MIN:
            break
        if (m - mins0) >= time_exit_min:
            timed_out = True
            break
        hi, lo = float(cache.high1[j]), float(cache.low1[j])
        hit_stop = lo <= stop_px if side > 0 else hi >= stop_px
        hit_tgt = hi >= target_px if side > 0 else lo <= target_px
        if hit_stop:                       # tie (both in one bar) -> stop
            return PathOutcome(j0, entry, j, stop_px, "stop", 0, -1.0)
        if hit_tgt:
            return PathOutcome(j0, entry, j, target_px, "target", 1,
                               k_target / k_stop)
        last = j
        j += 1
    # time / eod exit at the close of the last in-window bar
    exit_px = float(cache.close1[last])
    pnl_r = side * (exit_px - entry) / (k_stop * unit)
    return PathOutcome(j0, entry, last, exit_px,
                       "time" if timed_out else "eod",
                       1 if pnl_r > 0 else 0, float(pnl_r))


def outcomes_for_rows(cfg: Config, caches: dict[str, SymbolCache],
                      samples: Samples, rows: np.ndarray,
                      sides: np.ndarray) -> list[PathOutcome | None]:
    dl = cfg.raw["dl"]["trade"]
    by_id = {SYMBOLS_ORDER.index(s): c for s, c in caches.items()}
    out = []
    for r, side in zip(rows, sides):
        cache = by_id[int(samples.sym_id[r])]
        unit = float(samples.sigma_h[r]) * float(samples.close[r])
        out.append(walk_path(cache, int(samples.idx1[r]), float(side), unit,
                             float(dl["k_stop"]), float(dl["k_target"]),
                             int(dl["time_exit_min"])))
    return out


def net_pnl_r(cfg: Config, sym: str, entry_px: float, unit: float,
              pnl_r_gross: float, k_stop: float) -> float:
    """Charge measured round-trip cost, expressed in R (stop = 1R)."""
    comm = float(cfg.execution["commission_per_share"])
    c = costs.cost_r(costs.slip_frac(cfg, sym), comm, entry_px, k_stop, unit)
    return pnl_r_gross - c


def fit_calibrator_and_smin(cfg: Config, caches: dict[str, SymbolCache],
                            samples: Samples, val_rows: np.ndarray,
                            scores: np.ndarray):
    """VAL-ONLY: isotonic |s| -> P(target-before-stop); s_min by net expectancy
    over the pre-registered grid (min_trades, else default). Mirrors the
    _pick_threshold discipline."""
    from sklearn.isotonic import IsotonicRegression
    dl = cfg.raw["dl"]["trade"]
    k_stop = float(dl["k_stop"])
    sides = np.sign(scores)
    nz = sides != 0
    rows, s_abs, sides = val_rows[nz], np.abs(scores[nz]), sides[nz]
    outs = outcomes_for_rows(cfg, caches, samples, rows, sides)
    ok = np.array([o is not None for o in outs])
    rows, s_abs, sides = rows[ok], s_abs[ok], sides[ok]
    outs = [o for o in outs if o is not None]
    if len(outs) < 50:
        return None, float(dl["s_min_default"])
    wins = np.array([o.win for o in outs], dtype=float)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(s_abs, wins)

    net = np.array([net_pnl_r(cfg, SYMBOLS_ORDER[int(samples.sym_id[r])],
                              o.entry_px,
                              float(samples.sigma_h[r]) * float(samples.close[r]),
                              o.pnl_r_gross, k_stop)
                    for r, o in zip(rows, outs)])
    g = dl["s_min_grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 3)
    best, best_exp = float(dl["s_min_default"]), -np.inf
    for smin in grid:
        m = s_abs >= smin
        if m.sum() < int(dl["s_min_min_trades"]):
            continue
        e = float(net[m].mean())
        if e > best_exp:
            best_exp, best = e, float(smin)
    return calibrator, best


def build_trade_rows(cfg: Config, caches: dict[str, SymbolCache],
                     samples: Samples, rows: np.ndarray, scores: np.ndarray,
                     bundle, *, fold: str) -> list[dict]:
    """Blotter-schema rows for gated decisions (identical keys to the existing
    signals.json rows, setup='DL_SEQ'), with cooldown/concurrency hygiene."""
    dl = cfg.raw["dl"]["trade"]
    k_stop, k_target = float(dl["k_stop"]), float(dl["k_target"])
    fixed_shares = int(cfg.execution.get("fixed_shares", 1)) or 1
    comm = float(cfg.execution["commission_per_share"])
    s_min = float(bundle.s_min)

    order = rows[np.argsort(samples.ts[rows])]
    sc = {int(r): float(scores[i]) for i, r in enumerate(rows)}
    open_until: dict[int, np.datetime64] = {}
    cool_until: dict[int, np.datetime64] = {}
    n_open: list[tuple[np.datetime64, int]] = []
    out: list[dict] = []

    for r in order:
        s = sc[int(r)]
        side = 1.0 if s > 0 else -1.0
        sym_id = int(samples.sym_id[r])
        sym = SYMBOLS_ORDER[sym_id]
        ts = samples.ts[r]
        gated = abs(s) >= s_min
        p_cal = (float(bundle.calibrator.predict([abs(s)])[0])
                 if bundle.calibrator is not None else 0.5)
        unit = float(samples.sigma_h[r]) * float(samples.close[r])
        slip = costs.slip_frac(cfg, sym)
        cost = costs.cost_r(slip, comm, float(samples.close[r]), k_stop, unit)
        ev = costs.net_ev_r(p_cal, k_stop, k_target, cost)

        busy = (sym_id in open_until and ts < open_until[sym_id]) or \
               (sym_id in cool_until and ts < cool_until[sym_id])
        live_open = sum(1 for t, _ in n_open if t > ts)
        taken = bool(gated and ev > 0 and not busy
                     and live_open < int(dl["max_concurrent"]))

        outc = outcomes_for_rows(cfg, caches, samples,
                                 np.array([r]), np.array([side]))[0]
        if outc is None:
            continue
        cache = caches[sym]
        entry_ts = cache.ts1[outc.entry_idx]
        exit_ts = (cache.ts1[outc.exit_idx] + np.timedelta64(1, "m"))
        net_r = net_pnl_r(cfg, sym, outc.entry_px, unit, outc.pnl_r_gross, k_stop)
        # USD with slippage charged against the trade, commission both sides
        ef = outc.entry_px * (1.0 + side * slip)
        xf = outc.exit_px * (1.0 - side * slip)
        pnl_usd = (side * (xf - ef) - 2.0 * comm) * fixed_shares

        if taken:
            open_until[sym_id] = exit_ts
            cool_until[sym_id] = exit_ts + np.timedelta64(int(dl["cooldown_min"]), "m")
            n_open.append((exit_ts, sym_id))

        stop_px = outc.entry_px - side * k_stop * unit
        target_px = outc.entry_px + side * k_target * unit
        out.append({
            "id": int(r), "symbol": sym,
            "date": str(samples.session[r]),
            "setup": "DL_SEQ",
            "side": "LONG" if side > 0 else "SHORT",
            "trigger_ts": str(ts.astype("datetime64[s]")).replace(" ", "T"),
            "entry_ts": str(entry_ts.astype("datetime64[s]")).replace(" ", "T"),
            "entry_px": round(outc.entry_px, 2),
            "stop_px": round(stop_px, 2),
            "target_px": round(target_px, 2),
            "exit_ts": str(exit_ts.astype("datetime64[s]")).replace(" ", "T"),
            "exit_px": round(outc.exit_px, 2),
            "exit_reason": outc.exit_reason,
            "prob": round(p_cal, 3),
            "threshold": round(s_min, 3),
            "taken": taken,
            "outcome": "WIN" if net_r > 0 else "LOSS",
            "net_ev_r": round(float(ev), 4),
            "pnl_r": round(float(net_r), 3),
            "pnl_usd": round(float(pnl_usd), 2) if taken else None,
            "shares": fixed_shares if taken else None,
            "mae_r": 0.0, "mfe_r": 0.0,
            "fold": fold, "score": round(s, 4),
        })
    return out
