"""Performance statistics. Includes the anti-overfitting metrics that keep a
backtest honest: Probabilistic Sharpe Ratio and Deflated Sharpe Ratio
(Bailey & Lopez de Prado, 2014)."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

_EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation — accurate to ~1e-9, no scipy needed
    if not 0.0 < p < 1.0:
        raise ValueError("p out of range")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    sd = float(np.std(daily_returns, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(daily_returns) / sd * math.sqrt(252))


def probabilistic_sharpe(daily_returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true SR > benchmark) given skew/kurtosis-adjusted estimation error."""
    n = len(daily_returns)
    if n < 10:
        return 0.0
    sr = sharpe(daily_returns) / math.sqrt(252)          # per-period SR
    b = sr_benchmark / math.sqrt(252)
    m = np.mean(daily_returns)
    s = np.std(daily_returns, ddof=1)
    if s == 0:
        return 0.0
    z = (daily_returns - m) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    stat = (sr - b) * math.sqrt(n - 1) / denom
    return _norm_cdf(stat)


def deflated_sharpe(daily_returns: np.ndarray, n_trials: int) -> float:
    """DSR: PSR against the expected max SR from `n_trials` strategy variants.

    n_trials = geometry grid cells x threshold grid points actually searched.
    """
    n = len(daily_returns)
    if n < 10 or n_trials < 1:
        return 0.0
    srs = np.std(daily_returns, ddof=1)
    if srs == 0:
        return 0.0
    # variance of SR estimates across trials approximated by estimation variance
    var_sr = 1.0 / (n - 1)
    e_max = math.sqrt(var_sr) * (
        (1 - _EULER) * _norm_ppf(1 - 1.0 / n_trials)
        + _EULER * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    ) if n_trials > 1 else 0.0
    return probabilistic_sharpe(daily_returns, sr_benchmark=e_max * math.sqrt(252))


def max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min() * 100.0)


def summarize(trades: pl.DataFrame, equity_dates: list, equity: np.ndarray,
              n_trials: int) -> dict:
    if trades.height == 0:
        return {"trades": 0}
    pnl_usd = trades["pnl_usd"].to_numpy()
    pnl_r = trades["pnl_r"].to_numpy()
    wins = pnl_usd > 0
    losses = pnl_usd <= 0
    gross_win = float(pnl_usd[wins].sum()) if wins.any() else 0.0
    gross_loss = float(-pnl_usd[losses].sum()) if losses.any() else 0.0

    daily_ret = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])

    return {
        "trades": int(trades.height),
        "win_rate": round(float(wins.mean()) * 100.0, 2),
        "avg_win_usd": round(float(pnl_usd[wins].mean()), 2) if wins.any() else 0.0,
        "avg_loss_usd": round(float(pnl_usd[losses].mean()), 2) if losses.any() else 0.0,
        "avg_win_r": round(float(pnl_r[wins].mean()), 3) if wins.any() else 0.0,
        "avg_loss_r": round(float(pnl_r[losses].mean()), 3) if losses.any() else 0.0,
        "expectancy_r": round(float(pnl_r.mean()), 4),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else 0.0,
        "net_pnl_usd": round(float(pnl_usd.sum()), 2),
        "max_dd_pct": round(max_drawdown_pct(equity), 2),
        "sharpe": round(sharpe(daily_ret), 3),
        "psr": round(probabilistic_sharpe(daily_ret), 4),
        "deflated_sharpe": round(deflated_sharpe(daily_ret, n_trials), 4),
        "n_trials_deflation": n_trials,
    }
