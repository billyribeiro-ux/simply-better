"""Tests for the institutional layer: OOF purging, seed ensemble,
daily-loss kill switch, and bootstrap confidence intervals."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from engine import backtest, metrics, model
from engine.config import load_config
from engine.features import FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def _labeled(n_months=8, per_month=40, seed=5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        for k in range(per_month):
            d = date(2025, 1 + m, 1) + timedelta(days=int(k * 27 / per_month))
            row = {"session_date": d,
                   "entry_ts": datetime(d.year, d.month, d.day, 10, 0) + timedelta(minutes=k),
                   "outcome": int(rng.random() < 0.5),
                   "pnl_r": float(rng.normal(0, 1))}
            for f in FEATURES:
                row[f] = float(rng.normal(0, 1))
            rows.append(row)
    return pl.DataFrame(rows).sort("entry_ts")


# ---------------------------------------------------------------------------
# OOF purging: block k's model must never see the embargo window before it
# ---------------------------------------------------------------------------
class TestOofPurge:
    def test_fits_are_prior_only_with_embargo(self, cfg, monkeypatch):
        fits: list[int] = []

        class DummyClf:
            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

        def spy_fit(cfg_, X, y):
            fits.append(len(X))
            return [DummyClf()]

        monkeypatch.setattr(model, "_fit_clfs", spy_fit)
        train = _labeled()
        probs, valid = model._oof_probs(cfg, train)

        n = train.height
        K = int(cfg.model["oof_folds"])
        edges = [int(round(i * n / K)) for i in range(K + 1)]
        embargo = int(cfg.model["walk_forward"]["embargo_days"])
        dates = train["session_date"].to_list()

        assert not valid[: edges[1]].any()          # block 0 uncovered
        assert valid[edges[1]:].all()               # all later blocks covered
        assert len(fits) == K - 1
        for k, n_fit in zip(range(1, K), fits):
            block_start = dates[edges[k]]
            cutoff = block_start - timedelta(days=embargo)
            assert dates[n_fit - 1] < cutoff        # last fit row respects purge
            assert dates[n_fit] >= cutoff           # ...and is maximal


# ---------------------------------------------------------------------------
# seed ensemble
# ---------------------------------------------------------------------------
class TestEnsemble:
    def test_members_and_deterministic_scoring(self, cfg):
        train = _labeled()
        fm = model.fit_scored_model(cfg, train)
        assert len(fm.clfs) == int(cfg.model["ensemble_seeds"])
        rng = np.random.default_rng(9)
        X = rng.normal(0, 1, (16, len(FEATURES)))
        a, b = model.score(fm, X), model.score(fm, X)
        assert np.array_equal(a, b)
        assert ((a >= 0) & (a <= 1)).all()

    def test_oof_threshold_not_pinned_by_memorization(self, cfg):
        # with OOF probs the tail no longer reports AUC ~1.0
        fm = model.fit_scored_model(cfg, _labeled())
        assert fm.auc_val is not None
        assert fm.auc_val < 0.9      # random features: honest AUC ~0.5


# ---------------------------------------------------------------------------
# daily-loss kill switch
# ---------------------------------------------------------------------------
class TestKillSwitch:
    @staticmethod
    def _trade(day: date, h0: int, m0: int, h1: int, m1: int,
               entry: float, exit_: float) -> dict:
        return {
            "session_date": day,
            "entry_ts": datetime(day.year, day.month, day.day, h0, m0),
            "exit_ts": datetime(day.year, day.month, day.day, h1, m1),
            "side": 1.0, "stop_atr": 1.0, "atr": 4.0,
            "entry_px": entry, "exit_px": exit_,
        }

    def test_no_new_entries_after_daily_limit(self, cfg):
        d1, d2 = date(2026, 3, 2), date(2026, 3, 3)
        taken = pl.DataFrame([
            # -$20 x 125 shares = -2500 (-2.5%) -> breaches the 1.5% limit
            self._trade(d1, 10, 0, 10, 30, 100.0, 80.0),
            self._trade(d1, 11, 0, 11, 30, 100.0, 110.0),   # must be skipped
            self._trade(d2, 10, 0, 10, 30, 100.0, 110.0),   # next day trades again
        ])
        trades, dates, curve = backtest.run(cfg, taken)
        assert trades.height == 2
        assert trades["session_date"].to_list() == [d1, d2]
        assert float(trades["pnl_usd"][0]) < -2000

    def test_limit_disabled_when_zero(self, cfg):
        raw = dict(cfg.raw)
        raw["execution"] = {**raw["execution"], "max_daily_loss_pct": 0.0}
        cfg2 = type(cfg)(raw=raw, root=cfg.root)
        d1 = date(2026, 3, 2)
        taken = pl.DataFrame([
            self._trade(d1, 10, 0, 10, 30, 100.0, 80.0),
            self._trade(d1, 11, 0, 11, 30, 100.0, 110.0),
        ])
        trades, _, _ = backtest.run(cfg2, taken)
        assert trades.height == 2


# ---------------------------------------------------------------------------
# bootstrap confidence intervals
# ---------------------------------------------------------------------------
class TestBootstrapCI:
    def test_constant_series_degenerate_ci(self):
        lo, hi = metrics.bootstrap_ci(np.full(50, 0.25), np.mean)
        assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)

    def test_mean_ci_brackets_sample_mean(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0.1, 1.0, 400)
        lo, hi = metrics.bootstrap_ci(x, np.mean)
        assert lo < float(np.mean(x)) < hi
        assert hi - lo < 0.5

    def test_block_bootstrap_runs_and_brackets(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0.001, 0.01, 250)
        lo, hi = metrics.bootstrap_ci(x, metrics.sharpe, block=10)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi

    def test_summary_carries_cis(self):
        rng = np.random.default_rng(4)
        n = 60
        trades = pl.DataFrame({
            "pnl_usd": rng.normal(50, 400, n),
            "pnl_r": rng.normal(0.05, 1.0, n),
        })
        equity = 100000 * np.cumprod(1 + rng.normal(0.001, 0.01, 40))
        out = metrics.summarize(trades, [], np.concatenate(([100000.0], equity)), 100)
        lo, hi = out["expectancy_r_ci"]
        assert lo <= out["expectancy_r"] <= hi
        assert out["sharpe_ci"][0] <= out["sharpe"] <= out["sharpe_ci"][1]
