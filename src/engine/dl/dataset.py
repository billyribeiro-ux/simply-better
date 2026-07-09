"""Causal dataset builder — the no-lookahead surface of MIE-DL.

Every per-bar statistic uses information available at that bar's CLOSE and
nothing later, with the normalizing volatility taken from the PRIOR bar
(shift-then-use), trailing volume stats excluding the current bar, and
sequences never containing a bar whose close is after the decision time.
The same builder feeds training, evaluation, and live inference — parity by
construction. Pinned by tests/test_dl_dataset.py (future-perturbation:
mutating any bar after t must leave the tensors at t byte-identical).

Conventions (per the pre-registered spec, ledger 2026-07-09):
- bar timestamps are bar-OPEN times; a 1-min bar stamped 10:34 closes 10:35.
- decision bars sit on the 5-min grid `dl.decision.first..last`; the decision
  at time T uses the 1-min bar stamped T-1min (the last bar closing <= T).
- sigma_t: EWMA of squared within-session 1-min log returns (halflife
  `dl.ewma_halflife_1m`), value used at bar i is the EWMA through bar i-1.
- label: r_fwd = ln(close_{T+H} / close_T) normalized by sigma_t*sqrt(H);
  3-class deadband theta = round-trip cost (per-symbol measured slippage +
  commission) in the same vol units.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from .. import costs
from ..config import Config

log = logging.getLogger("engine.dl.dataset")

SYMBOLS_ORDER = ["AAPL", "NVDA", "AMZN", "NFLX", "SPY", "QQQ", "IWM", "TSLA"]
N_CH_1M = 9
N_CH_5M = 6


# ---------------------------------------------------------------------------
# causal per-bar statistics
# ---------------------------------------------------------------------------
def ewma_sigma_prior(r: np.ndarray, new_session: np.ndarray,
                     halflife: float, warmup: int) -> np.ndarray:
    """sigma[i] = sqrt(EWMA of r^2 through bar i-1), within-session returns
    only (the overnight return never enters). First `warmup` bars get NaN
    (dropped by callers). Strictly causal by construction."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n = len(r)
    out = np.full(n, np.nan)
    ew = np.nan
    for i in range(n):
        out[i] = np.sqrt(ew) if np.isfinite(ew) else np.nan
        if not new_session[i]:                     # within-session return only
            ri2 = r[i] * r[i]
            ew = ri2 if not np.isfinite(ew) else (1 - alpha) * ew + alpha * ri2
    out[:warmup] = np.nan
    return out


def trailing_z(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """(x[i] - mean(x[i-window:i])) / std(...), current bar EXCLUDED."""
    n = len(x)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    csum2 = np.concatenate([[0.0], np.cumsum(x * x)])
    out = np.full(n, np.nan)
    for i in range(min_periods, n):
        lo = max(0, i - window)
        cnt = i - lo
        m = (csum[i] - csum[lo]) / cnt
        v = (csum2[i] - csum2[lo]) / cnt - m * m
        s = np.sqrt(max(v, 1e-12))
        out[i] = (x[i] - m) / s
    return out


# ---------------------------------------------------------------------------
# per-symbol caches
# ---------------------------------------------------------------------------
@dataclass
class SymbolCache:
    symbol: str
    # 1-min level (aligned to that symbol's own 1-min bars, RTH only)
    ts1: np.ndarray          # datetime64[us] bar-open
    date1: np.ndarray        # datetime64[D]
    open1: np.ndarray        # raw OHLC retained for the trade-path walk
    high1: np.ndarray
    low1: np.ndarray
    close1: np.ndarray
    feat1: np.ndarray        # (N1, 9) float32 — channel order per module doc
    sigma1: np.ndarray       # prior-EWMA sigma per bar (1-min)
    # 5-min level
    ts5: np.ndarray
    feat5: np.ndarray        # (N5, 6) float32
    valid1: np.ndarray       # bool — all features finite at this bar


def _rth(df: pl.DataFrame) -> pl.DataFrame:
    t = pl.col("ts")
    mins = t.dt.hour().cast(pl.Int32) * 60 + t.dt.minute().cast(pl.Int32)
    return df.filter((mins >= 570) & (mins < 960)).sort("ts")   # 09:30..16:00


def _mkt_aligned_r(base_ts: np.ndarray, mkt: pl.DataFrame,
                   halflife: float, warmup: int) -> np.ndarray:
    """Market (SPY/QQQ) 1-min normalized return aligned onto base_ts by exact
    bar-open timestamp; missing minutes = 0.0 (no-trade minute)."""
    ts = mkt["ts"].to_numpy()
    cl = mkt["close"].to_numpy().astype(float)
    r = np.concatenate([[0.0], np.diff(np.log(cl))])
    dts = mkt["ts"].dt.date().to_numpy()
    new = np.concatenate([[True], dts[1:] != dts[:-1]])
    r[new] = 0.0
    sig = ewma_sigma_prior(r, new, halflife, warmup)
    rn = np.where(np.isfinite(sig) & (sig > 0), r / np.where(sig > 0, sig, 1), 0.0)
    rn = np.clip(rn, -8, 8)
    idx = {int(t): i for i, t in enumerate(ts.astype("datetime64[us]").astype(np.int64))}
    out = np.zeros(len(base_ts), dtype=np.float64)
    base_i = base_ts.astype("datetime64[us]").astype(np.int64)
    for j, key in enumerate(base_i):
        i = idx.get(int(key))
        if i is not None and np.isfinite(rn[i]):
            out[j] = rn[i]
    return out


def build_symbol_cache(cfg: Config, symbol: str,
                       bars1: pl.DataFrame, bars5: pl.DataFrame,
                       spy1: pl.DataFrame, qqq1: pl.DataFrame) -> SymbolCache:
    dl = cfg.raw["dl"]
    hl1, hl5 = float(dl["ewma_halflife_1m"]), float(dl["ewma_halflife_5m"])
    b1, b5 = _rth(bars1), _rth(bars5)

    # ---- 1-min ----
    ts1 = b1["ts"].to_numpy()
    o = b1["open"].to_numpy().astype(float)
    h = b1["high"].to_numpy().astype(float)
    lo = b1["low"].to_numpy().astype(float)
    c = b1["close"].to_numpy().astype(float)
    v = b1["volume"].to_numpy().astype(float)
    dts = b1["ts"].dt.date().to_numpy()
    new = np.concatenate([[True], dts[1:] != dts[:-1]])

    r = np.concatenate([[0.0], np.diff(np.log(c))])
    r[new] = 0.0
    sigma = ewma_sigma_prior(r, new, hl1, warmup=390)
    safe_sig = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)

    r_norm = np.clip(r / safe_sig, -8, 8)
    rng_ = np.log(np.maximum(h, 1e-9) / np.maximum(lo, 1e-9))
    range_norm = np.clip(rng_ / safe_sig, 0, 16)
    span = h - lo
    close_loc = np.where(span > 0, (c - lo) / np.where(span > 0, span, 1) - 0.5, 0.0)
    vol_z = np.clip(trailing_z(np.log1p(v), 390, 100), -6, 6)

    mins = (b1["ts"].dt.hour().cast(pl.Int32) * 60
            + b1["ts"].dt.minute().cast(pl.Int32)).to_numpy() - 570
    tod = 2 * np.pi * mins / 390.0
    t_close = (385 - mins) / 385.0                 # minutes until 15:55

    spy_ch = (np.zeros(len(ts1)) if symbol == "SPY"
              else _mkt_aligned_r(ts1, _rth(spy1), hl1, 390))
    qqq_ch = (np.zeros(len(ts1)) if symbol == "QQQ"
              else _mkt_aligned_r(ts1, _rth(qqq1), hl1, 390))

    feat1 = np.stack([r_norm, range_norm, close_loc, vol_z, spy_ch, qqq_ch,
                      np.sin(tod), np.cos(tod), t_close], axis=1)
    valid1 = np.isfinite(feat1).all(axis=1)
    feat1 = np.nan_to_num(feat1, nan=0.0).astype(np.float32)

    # ---- 5-min ----
    ts5 = b5["ts"].to_numpy()
    c5 = b5["close"].to_numpy().astype(float)
    h5 = b5["high"].to_numpy().astype(float)
    l5 = b5["low"].to_numpy().astype(float)
    v5 = b5["volume"].to_numpy().astype(float)
    d5 = b5["ts"].dt.date().to_numpy()
    new5 = np.concatenate([[True], d5[1:] != d5[:-1]])
    r5 = np.concatenate([[0.0], np.diff(np.log(c5))])
    # overnight gap: keep (context may span sessions) but clip at +-5 sigma
    sig5 = ewma_sigma_prior(np.where(new5, 0.0, r5), new5, hl5, warmup=78)
    ss5 = np.where(np.isfinite(sig5) & (sig5 > 0), sig5, np.nan)
    r5n = r5 / ss5
    r5n = np.where(new5, np.clip(r5n, -5, 5), np.clip(r5n, -8, 8))
    rng5 = np.clip(np.log(np.maximum(h5, 1e-9) / np.maximum(l5, 1e-9)) / ss5, 0, 16)
    span5 = h5 - l5
    cl5 = np.where(span5 > 0, (c5 - l5) / np.where(span5 > 0, span5, 1) - 0.5, 0.0)
    vz5 = np.clip(trailing_z(np.log1p(v5), 78, 30), -6, 6)
    m5 = (b5["ts"].dt.hour().cast(pl.Int32) * 60
          + b5["ts"].dt.minute().cast(pl.Int32)).to_numpy() - 570
    feat5 = np.stack([r5n, rng5, cl5, vz5, new5.astype(float),
                      np.sin(2 * np.pi * m5 / 390.0)], axis=1)
    feat5 = np.nan_to_num(feat5, nan=0.0).astype(np.float32)

    return SymbolCache(symbol=symbol, ts1=ts1, date1=dts,
                       open1=o, high1=h, low1=lo, close1=c,
                       feat1=feat1, sigma1=sigma, ts5=ts5, feat5=feat5,
                       valid1=valid1)


def build_caches(cfg: Config, symbols: list[str] | None = None) -> dict[str, SymbolCache]:
    """Build (or load from data/dl) causal caches for all symbols."""
    from ..data import Store
    store = Store(cfg)
    symbols = symbols or [s for s in SYMBOLS_ORDER if s in cfg.universe]
    far, today = date(2000, 1, 1), date(2100, 1, 1)
    spy1 = store.load_bars("1min", "SPY", far, today)
    qqq1 = store.load_bars("1min", "QQQ", far, today)
    out: dict[str, SymbolCache] = {}
    for sym in symbols:
        b1 = store.load_bars("1min", sym, far, today)
        b5 = store.load_bars("5min", sym, far, today)
        out[sym] = build_symbol_cache(cfg, sym, b1, b5, spy1, qqq1)
        log.info("[%s] cache: %d 1m bars, %d 5m bars", sym,
                 len(out[sym].ts1), len(out[sym].ts5))
    return out


# ---------------------------------------------------------------------------
# decision samples
# ---------------------------------------------------------------------------
@dataclass
class Samples:
    """Flat arrays; one row per (symbol, decision bar)."""
    sym_id: np.ndarray       # int64 into SYMBOLS_ORDER
    idx1: np.ndarray         # decision-bar index into the symbol's 1-min arrays
    idx5: np.ndarray         # last completed 5-min bar index
    ts: np.ndarray           # decision time (datetime64[us]) = bar close
    session: np.ndarray      # datetime64[D]
    close: np.ndarray        # close at decision
    sigma_h: np.ndarray      # sigma_t * sqrt(H)  (label + trade vol unit)
    y_ret: np.ndarray        # normalized H-forward return
    y_cls: np.ndarray        # 0 down / 1 flat / 2 up
    theta: np.ndarray        # cost deadband in vol units
    dow_sin: np.ndarray
    dow_cos: np.ndarray


def _hm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def build_samples(cfg: Config, caches: dict[str, SymbolCache],
                  d_from: date, d_to: date,
                  horizon_min: int | None = None,
                  require_label: bool = True) -> Samples:
    """require_label=False (live path): decision bars near 'now' whose label
    horizon has not elapsed are still emitted with y_ret/y_cls = 0 — scoring
    needs only the input window, never the label."""
    dl = cfg.raw["dl"]
    H = int(horizon_min or dl["horizon_min"])
    L1 = int(dl["seq_1m"])
    comm = float(cfg.execution["commission_per_share"])
    first = _hm_to_min(dl["decision"]["first"])
    last = _hm_to_min(dl["decision"]["last"])

    cols: dict[str, list] = {k: [] for k in
        ("sym_id", "idx1", "idx5", "ts", "session", "close", "sigma_h",
         "y_ret", "y_cls", "theta", "dow_sin", "dow_cos")}

    for sym, cache in caches.items():
        sid = SYMBOLS_ORDER.index(sym)
        slip = costs.slip_frac(cfg, sym)
        n1 = len(cache.ts1)
        mins = ((cache.ts1.astype("datetime64[m]")
                 - cache.ts1.astype("datetime64[D]")).astype(int))
        t5_close = (cache.ts5 + np.timedelta64(5, "m")).astype("datetime64[us]")
        dts64 = cache.date1.astype("datetime64[D]")
        lo64 = np.datetime64(d_from, "D")
        hi64 = np.datetime64(d_to, "D")

        # session start indices for the within-session window check
        new = np.concatenate([[True], cache.date1[1:] != cache.date1[:-1]])
        sess_start = np.zeros(n1, dtype=np.int64)
        cur = 0
        for i in range(n1):
            if new[i]:
                cur = i
            sess_start[i] = cur

        for i in range(n1):
            if not (lo64 <= dts64[i] <= hi64):
                continue
            m = mins[i] + 1                       # decision time = bar close
            if m < first or m > last or (m - 570) % 5 != 0:
                continue
            # need L1 within-session completed bars ending at i
            if i - sess_start[i] + 1 < L1:
                continue
            # label: H later 1-min bars in the SAME session
            j = i + H
            has_label = j < n1 and cache.date1[j] == cache.date1[i]
            if require_label and not has_label:
                continue
            sig = cache.sigma1[i]
            if not (np.isfinite(sig) and sig > 0) or not cache.valid1[i]:
                continue
            # last completed 5-min bar (close <= decision time)
            dec_ts = (cache.ts1[i] + np.timedelta64(1, "m")).astype("datetime64[us]")
            k5 = int(np.searchsorted(t5_close, dec_ts, side="right")) - 1
            if k5 < int(dl["seq_5m"]) - 1:
                continue

            sig_h = sig * np.sqrt(H)
            c_rt = 2.0 * (slip + comm / cache.close1[i])
            th = c_rt / sig_h
            if has_label:
                r_fwd = np.log(cache.close1[j] / cache.close1[i])
                y = r_fwd / sig_h
                cls = 2 if y > th else (0 if y < -th else 1)
            else:
                y, cls = 0.0, 1

            wd = int(dts64[i].astype("datetime64[D]").astype(object).weekday())
            cols["sym_id"].append(sid); cols["idx1"].append(i)
            cols["idx5"].append(k5); cols["ts"].append(dec_ts)
            cols["session"].append(dts64[i]); cols["close"].append(cache.close1[i])
            cols["sigma_h"].append(sig_h); cols["y_ret"].append(y)
            cols["y_cls"].append(cls); cols["theta"].append(th)
            cols["dow_sin"].append(np.sin(2 * np.pi * wd / 5.0))
            cols["dow_cos"].append(np.cos(2 * np.pi * wd / 5.0))

    return Samples(
        sym_id=np.array(cols["sym_id"], dtype=np.int64),
        idx1=np.array(cols["idx1"], dtype=np.int64),
        idx5=np.array(cols["idx5"], dtype=np.int64),
        ts=np.array(cols["ts"], dtype="datetime64[us]"),
        session=np.array(cols["session"], dtype="datetime64[D]"),
        close=np.array(cols["close"], dtype=np.float64),
        sigma_h=np.array(cols["sigma_h"], dtype=np.float64),
        y_ret=np.array(cols["y_ret"], dtype=np.float32),
        y_cls=np.array(cols["y_cls"], dtype=np.int64),
        theta=np.array(cols["theta"], dtype=np.float32),
        dow_sin=np.array(cols["dow_sin"], dtype=np.float32),
        dow_cos=np.array(cols["dow_cos"], dtype=np.float32),
    )


def tensor_batch(cfg: Config, caches: dict[str, SymbolCache],
                 samples: Samples, rows: np.ndarray):
    """Materialize (x1m, x5m, sym, stat) torch tensors for `rows`."""
    torch = __import__("torch")
    dl = cfg.raw["dl"]
    L1, L5 = int(dl["seq_1m"]), int(dl["seq_5m"])
    by_sym: dict[int, SymbolCache] = {
        SYMBOLS_ORDER.index(s): c for s, c in caches.items()}
    B = len(rows)
    x1 = np.empty((B, N_CH_1M, L1), dtype=np.float32)
    x5 = np.empty((B, N_CH_5M, L5), dtype=np.float32)
    for b, ridx in enumerate(rows):
        c = by_sym[int(samples.sym_id[ridx])]
        i, k5 = int(samples.idx1[ridx]), int(samples.idx5[ridx])
        x1[b] = c.feat1[i - L1 + 1: i + 1].T
        x5[b] = c.feat5[k5 - L5 + 1: k5 + 1].T
    stat = np.stack([samples.dow_sin[rows], samples.dow_cos[rows]], axis=1)
    return (torch.from_numpy(x1), torch.from_numpy(x5),
            torch.from_numpy(samples.sym_id[rows]),
            torch.from_numpy(stat.astype(np.float32)))
