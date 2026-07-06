"""Synthetic end-to-end smoke test: 300 sessions of fake 5m/1m/daily bars
for 2 symbols through the entire pipeline."""
import numpy as np, polars as pl
from datetime import date, datetime, timedelta
from engine.config import load_config
from engine import learn
from engine.data import Store

rng = np.random.default_rng(7)
cfg = load_config("config.yaml")

def gen_symbol(sym, start, n_days, px0):
    daily_rows, b5, b1 = [], [], []
    px, d, made = px0, start, 0
    while made < n_days:
        if d.weekday() >= 5:
            d += timedelta(days=1); continue
        o = px * (1 + rng.normal(0, 0.004))
        mins = 390
        rets = rng.normal(0, 0.0009, mins)
        drift = np.sin(np.linspace(0, np.pi * rng.uniform(1, 3), mins)) * rng.uniform(0.002, 0.012)
        closes = o * np.cumprod(1 + rets) * (1 + drift)
        highs = closes * (1 + np.abs(rng.normal(0, 0.0006, mins)))
        lows = closes * (1 - np.abs(rng.normal(0, 0.0006, mins)))
        opens = np.concatenate(([o], closes[:-1]))
        vols = rng.uniform(1e4, 6e4, mins)
        t0 = datetime(d.year, d.month, d.day, 9, 30)
        for i in range(mins):
            b1.append((t0 + timedelta(minutes=i), float(opens[i]), float(max(opens[i], highs[i])),
                       float(min(opens[i], lows[i])), float(closes[i]), float(vols[i]), sym))
        for i in range(0, mins, 5):
            sl = slice(i, i + 5)
            b5.append((t0 + timedelta(minutes=i), float(opens[i]), float(highs[sl].max()),
                       float(lows[sl].min()), float(closes[i + 4]), float(vols[sl].sum()), sym))
        daily_rows.append((d, float(o), float(highs.max()), float(lows.min()),
                           float(closes[-1]), float(vols.sum())))
        px = closes[-1]; made += 1; d += timedelta(days=1)
    schema5 = ["ts", "open", "high", "low", "close", "volume", "symbol"]
    return (pl.DataFrame(daily_rows, schema=["date", "open", "high", "low", "close", "volume"], orient="row"),
            pl.DataFrame(b5, schema=schema5, orient="row"),
            pl.DataFrame(b1, schema=schema5, orient="row"))

store = Store(cfg)
start = date(2024, 1, 2)
for sym, px0 in [("AAPL", 190.0), ("TSLA", 240.0)]:
    daily, b5, b1 = gen_symbol(sym, start, 300, px0)
    store.save_daily(sym, daily)
    store.save_bars("5min", sym, b5)
    store.save_bars("1min", sym, b1)

cfg.raw["universe"] = ["AAPL", "TSLA"]
art = learn.run_pipeline(cfg, start, date(2025, 4, 30))
k = art["summary"]
print("SMOKE:", {x: k.get(x) for x in ("trades", "win_rate", "expectancy_r",
                                        "net_pnl_usd", "sharpe", "deflated_sharpe")})
assert art["folds"] >= 3 and len(art["signals"]) > 0
print("SMOKE TEST PASS")
