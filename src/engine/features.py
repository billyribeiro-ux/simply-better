"""Daily context features used by the setup detector and the model.

Everything here is strictly backward-looking: row d only sees data < d.
"""
from __future__ import annotations

import numpy as np
import polars as pl

FEATURES: list[str] = [
    "side",
    "minutes_since_open",
    "tod_sin",
    "tod_cos",
    "gap_pct",
    "range_ext_atr",
    "vol_z",
    "dist_vwap_atr",
    "prior_high_dist_atr",
    "prior_low_dist_atr",
    "gk_vol_z",
    "ret_30m_atr",
    "wick_ratio",
    "n_extremes",
    "dow",
    "atr_pct",
    # iteration two: anchor framework — appended, never reordered
    "room_vwap_atr",
    "room_open_atr",
    "room_pclose_atr",
    "room_pdpoc_atr",
    "room_dpoc_atr",
    "vwap_slope_atr",
    "max_room_atr",
]


def daily_context(daily: pl.DataFrame) -> pl.DataFrame:
    """ATR14, GK vol z (60d), prior-day OHLC — all shifted so nothing leaks."""
    if daily.height == 0:
        return daily
    df = daily.sort("date")
    o = df["open"].to_numpy().astype(float)
    h = df["high"].to_numpy().astype(float)
    lo = df["low"].to_numpy().astype(float)
    c = df["close"].to_numpy().astype(float)

    prev_c = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    atr14 = _roll_mean(tr, 14)

    with np.errstate(divide="ignore", invalid="ignore"):
        gk = np.sqrt(
            np.maximum(
                0.5 * np.log(h / lo) ** 2
                - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2,
                0.0,
            )
        )
    gk_mu = _roll_mean(gk, 60)
    gk_sd = _roll_std(gk, 60)
    gk_z = np.where(gk_sd > 0, (gk - gk_mu) / gk_sd, 0.0)

    out = df.with_columns(
        pl.Series("atr14", atr14),
        pl.Series("gk_z", gk_z),
    ).with_columns(
        # everything an intraday session may consult must come from the PRIOR day
        pl.col("atr14").shift(1).alias("atr"),
        pl.col("gk_z").shift(1).alias("gk_vol_z"),
        pl.col("high").shift(1).alias("prior_high"),
        pl.col("low").shift(1).alias("prior_low"),
        pl.col("close").shift(1).alias("prior_close"),
    )
    return out.select("date", "atr", "gk_vol_z", "prior_high", "prior_low", "prior_close")


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    csum = np.nancumsum(np.where(np.isnan(x), 0.0, x))
    cnt = np.cumsum(~np.isnan(x))
    for i in range(len(x)):
        j = max(0, i - n + 1)
        k = cnt[i] - (cnt[j - 1] if j > 0 else 0)
        if k >= max(2, n // 2):
            s = csum[i] - (csum[j - 1] if j > 0 else 0.0)
            out[i] = s / k
    return out


def _roll_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        j = max(0, i - n + 1)
        w = x[j : i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= max(2, n // 2):
            out[i] = float(np.std(w, ddof=1))
    return out
