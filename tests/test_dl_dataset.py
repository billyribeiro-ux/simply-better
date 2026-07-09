"""MIE-DL dataset causality tests — the no-lookahead surface.

The decisive test: mutate every bar strictly AFTER decision bar t and require
the tensors at t to be byte-identical. If any future information leaks into
features, this fails.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from engine.config import load_config          # noqa: E402
from engine.dl import dataset as ds            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def _mk_bars(seed: int, n_days: int = 8, start: date = date(2025, 3, 3)):
    """Synthetic 1-min + 5-min RTH bars with realistic scale."""
    rng = np.random.default_rng(seed)
    rows1, rows5 = [], []
    px = 100.0
    d = start
    made = 0
    while made < n_days:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        for m in range(390):
            ts = datetime(d.year, d.month, d.day, 9, 30) + timedelta(minutes=m)
            r = rng.normal(0, 0.0008)
            o = px
            px = px * (1 + r)
            hi = max(o, px) * (1 + abs(rng.normal(0, 0.0002)))
            lo_ = min(o, px) * (1 - abs(rng.normal(0, 0.0002)))
            rows1.append((ts, o, hi, lo_, px, float(rng.integers(500, 5000))))
        made += 1
        d += timedelta(days=1)
    b1 = pl.DataFrame(rows1, schema=["ts", "open", "high", "low", "close",
                                     "volume"], orient="row")
    # aggregate 5-min from 1-min (consistent by construction)
    b5 = (b1.with_columns(
            ((pl.col("ts").dt.hour().cast(pl.Int64) * 60
              + pl.col("ts").dt.minute().cast(pl.Int64)) // 5).alias("bkt"),
            pl.col("ts").dt.date().alias("d"))
          .group_by(["d", "bkt"], maintain_order=True)
          .agg(pl.col("ts").first(), pl.col("open").first(),
               pl.col("high").max(), pl.col("low").min(),
               pl.col("close").last(), pl.col("volume").sum())
          .drop(["d", "bkt"]))
    return b1, b5


def _cache(cfg, seed=1, n_days=8):
    b1, b5 = _mk_bars(seed, n_days)
    spy1, _ = _mk_bars(seed + 100, n_days)
    qqq1, _ = _mk_bars(seed + 200, n_days)
    # shrink warmup for the synthetic set: patch via smaller history is fine —
    # the builder drops the first 390 bars (one synthetic session)
    return ds.build_symbol_cache(cfg, "AAPL", b1, b5, spy1, qqq1), (b1, b5, spy1, qqq1)


class TestCausality:
    def test_future_perturbation_leaves_tensors_identical(self, cfg):
        cache, (b1, b5, spy1, qqq1) = _cache(cfg)
        samples = ds.build_samples(cfg, {"AAPL": cache},
                                   date(2025, 1, 1), date(2025, 12, 31))
        assert samples.y_cls.size > 50, "synthetic set produced no samples"
        # pick a mid-sample; mutate ALL bars strictly after its decision time
        k = samples.y_cls.size // 2
        dec_ts = samples.ts[k]
        cut = np.datetime64(dec_ts)

        def poison(df):
            ts = df["ts"].to_numpy().astype("datetime64[us]")
            fut = ts >= cut          # decision bar itself stays (closes at cut)
            f = df.to_pandas()
            for col in ("open", "high", "low", "close", "volume"):
                vals = f[col].to_numpy(copy=True)
                vals[fut] = vals[fut] * 7.7 + 11.1
                f[col] = vals
            return pl.DataFrame(f)

        cache2 = ds.build_symbol_cache(cfg, "AAPL", poison(b1), poison(b5),
                                       poison(spy1), poison(qqq1))
        samples2 = ds.build_samples(cfg, {"AAPL": cache2},
                                    date(2025, 1, 1), date(2025, 12, 31))
        # the SAME decision bar must exist and its input tensors must match
        m2 = np.where(samples2.ts == dec_ts)[0]
        assert m2.size == 1
        x1a, x5a, sya, sta = ds.tensor_batch(cfg, {"AAPL": cache}, samples,
                                             np.array([k]))
        x1b, x5b, syb, stb = ds.tensor_batch(cfg, {"AAPL": cache2}, samples2,
                                             m2)
        assert torch.equal(x1a, x1b), "1-min tensor changed when the future changed"
        assert torch.equal(x5a, x5b), "5-min tensor changed when the future changed"
        assert torch.equal(sta, stb)

    def test_sigma_is_prior_bar_only(self):
        # constant returns then a huge shock: sigma at the shock bar must not
        # yet include the shock (shift-then-use)
        r = np.full(600, 1e-4)
        r[500] = 0.05
        new = np.zeros(600, dtype=bool); new[0] = True
        sig = ds.ewma_sigma_prior(r, new, halflife=120, warmup=390)
        assert np.isnan(sig[:390]).all()
        assert sig[500] < 0.001            # shock NOT visible at its own bar
        assert sig[501] > sig[500] * 2     # visible one bar later

    def test_trailing_z_excludes_current_bar(self):
        x = np.zeros(300); x[250] = 100.0
        z = ds.trailing_z(x, window=100, min_periods=50)
        # at the spike bar, mean/std come from the flat past -> huge z;
        # the spike itself must not deflate its own score via the window
        assert z[250] > 50
        assert abs(z[249]) < 1e-6

    def test_deadband_class_math(self, cfg):
        cache, _ = _cache(cfg, seed=3)
        s = ds.build_samples(cfg, {"AAPL": cache},
                             date(2025, 1, 1), date(2025, 12, 31))
        up = s.y_cls == 2
        dn = s.y_cls == 0
        fl = s.y_cls == 1
        assert (s.y_ret[up] > s.theta[up]).all()
        assert (s.y_ret[dn] < -s.theta[dn]).all()
        assert (np.abs(s.y_ret[fl]) <= s.theta[fl] + 1e-12).all()
        assert (s.theta > 0).all()

    def test_decision_grid_and_window(self, cfg):
        cache, _ = _cache(cfg, seed=4)
        s = ds.build_samples(cfg, {"AAPL": cache},
                             date(2025, 1, 1), date(2025, 12, 31))
        mins = ((s.ts.astype("datetime64[m]")
                 - s.ts.astype("datetime64[D]")).astype(int))
        assert ((mins - 570) % 5 == 0).all()
        assert mins.min() >= 635 and mins.max() <= 930      # 10:35..15:30
