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
    c = load_config(REPO_ROOT / "config.yaml")
    # these tests exercise the fade detectors; re-enable them (production
    # config disables the refuted concept)
    for s in ("hod_fade", "lod_reclaim"):
        c.raw["setups"][s]["enabled"] = True
    return c


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


# ---------------------------------------------------------------------------
# slippage direction (invariant 7: charged AGAINST the trade)
# ---------------------------------------------------------------------------
class TestSlippageDirection:
    def test_long_and_short_both_pay(self, cfg):
        d = date(2026, 3, 2)

        def scratch_trade(side: float) -> pl.DataFrame:
            # entry == exit: gross must be NEGATIVE once slippage is charged
            return pl.DataFrame([{
                "session_date": d,
                "entry_ts": datetime(2026, 3, 2, 10, 0),
                "exit_ts": datetime(2026, 3, 2, 11, 0),
                "side": side, "stop_atr": 1.0, "atr": 4.0,
                "entry_px": 100.0, "exit_px": 100.0,
            }])

        for side in (1.0, -1.0):
            trades, _, _ = backtest.run(cfg, scratch_trade(side))
            assert trades.height == 1
            assert float(trades["pnl_usd"][0]) < 0, f"side={side} was not charged"


# ---------------------------------------------------------------------------
# day-type trend veto plumbing
# ---------------------------------------------------------------------------
def _lod_eff_expected(bars):
    cl = [b[3] for b in bars]
    path = abs(cl[0] - 100.0) + sum(abs(cl[k] - cl[k - 1]) for k in range(1, 8))
    return (cl[7] - 100.0) / path


class TestTrendVeto:
    def test_setups_emit_aligned_efficiency(self, cfg):
        # crafted up-off-the-open day: HOD_FADE fights the move (positive
        # aligned efficiency), LOD_RECLAIM rides it (negative)
        from engine import setups
        day = date(2025, 3, 10)
        bars = [
            (100.00, 100.20, 99.80, 100.10, 1000.0),
            (100.10, 100.30, 99.90, 100.00, 1200.0),
            (100.00, 100.10, 99.70, 99.90, 900.0),
            (99.99, 100.02, 100.00, 100.01, 50000.0),
            (100.01, 100.25, 99.85, 100.05, 1100.0),
            (100.05, 100.30, 99.90, 100.15, 1000.0),
            (101.20, 102.00, 100.90, 101.00, 8000.0),   # HOD_FADE @10:00
            (99.30, 99.70, 99.00, 99.60, 800.0),        # LOD_RECLAIM @10:05
            (99.60, 99.80, 99.40, 99.70, 900.0),
            (99.70, 99.90, 99.50, 99.80, 900.0),
        ]
        t0 = datetime(2025, 3, 10, 9, 30)
        b5 = pl.DataFrame(
            [(t0 + timedelta(minutes=5 * i), o, h, lo, c, v, "TEST")
             for i, (o, h, lo, c, v) in enumerate(bars)],
            schema=["ts", "open", "high", "low", "close", "volume", "symbol"],
            orient="row")
        ctx = pl.DataFrame({
            "date": [day], "atr": [2.0], "gk_vol_z": [0.0],
            "prior_high": [100.5], "prior_low": [99.5], "prior_close": [100.0],
        })
        events = setups.detect(cfg, "TEST", b5, ctx)
        by = {e["setup"]: e for e in events}
        hod, lod = by["HOD_FADE"], by["LOD_RECLAIM"]
        assert hod["dt_eff_h1_aligned"] > 0
        # independently recompute eff at the HOD trigger bar (bars 0..6)
        cl = [b[3] for b in bars]
        path = abs(cl[0] - 100.0) + sum(abs(cl[k] - cl[k - 1]) for k in range(1, 7))
        eff = (cl[6] - 100.0) / path
        assert hod["dt_eff_h1_aligned"] == pytest.approx(eff)
        assert lod["dt_eff_h1_aligned"] == pytest.approx(-_lod_eff_expected(bars))


# ---------------------------------------------------------------------------
# PathScan research instrumentation
# ---------------------------------------------------------------------------
class TestPathInstrumentation:
    def test_paths_stored_and_consistent(self, cfg):
        from engine import labeling
        day = date(2025, 3, 10)
        t0 = datetime(2025, 3, 10, 10, 0)
        ev = {
            "symbol": "TEST", "session_date": day, "trigger_ts": t0,
            "trigger_low": 100.0, "trigger_high": 101.0,
            "side": -1.0, "atr": 2.0,
        }
        t1 = datetime(2025, 3, 10, 10, 5)
        b1 = pl.DataFrame(
            [(t1 + timedelta(minutes=i), 99.9, 100.1, 99.5 - 0.02 * i,
              99.8 - 0.02 * i, 500.0, "TEST") for i in range(30)],
            schema=["ts", "open", "high", "low", "close", "volume", "symbol"],
            orient="row")
        scans = labeling.scan_paths(cfg, [ev], {"TEST": b1})
        assert 0 in scans
        s = scans[0]
        n = len(s.cross_ts)
        assert s.fav_path is not None and len(s.fav_path) == n
        assert s.adv_path is not None and len(s.adv_path) == n
        assert s.close_path is not None and len(s.close_path) == n
        assert float(s.close_path[-1]) == pytest.approx(s.eod_px, abs=1e-4)
        # signed eod excursion consistent with the stored close path (short)
        assert s.eod_signed_atr == pytest.approx(
            (s.entry_px - float(s.close_path[-1])) / 2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# period report builder
# ---------------------------------------------------------------------------
class TestPeriodReport:
    ART = {
        "run_id": "testrun",
        "signals": [
            {"id": 1, "symbol": "NVDA", "date": "2026-05-04", "setup": "HOD_BREAK",
             "side": "LONG", "trigger_ts": "2026-05-04T10:30:00",
             "entry_ts": "2026-05-04T10:36:00", "entry_px": 100.0,
             "stop_px": 99.0, "target_px": 102.0,
             "exit_ts": "2026-05-04T11:10:00", "exit_px": 102.0,
             "exit_reason": "target", "prob": 0.61, "threshold": 0.55,
             "taken": True, "outcome": "WIN", "pnl_r": 2.0,
             "pnl_usd": 500.0, "shares": 250, "mae_r": 0.2, "mfe_r": 2.0},
            {"id": 2, "symbol": "TSLA", "date": "2026-05-05", "setup": "LOD_BREAK",
             "side": "SHORT", "trigger_ts": "2026-05-05T11:00:00",
             "entry_ts": "2026-05-05T11:04:00", "entry_px": 300.0,
             "stop_px": 303.0, "target_px": 294.0,
             "exit_ts": "2026-05-05T12:00:00", "exit_px": 303.0,
             "exit_reason": "stop", "prob": 0.40, "threshold": 0.55,
             "taken": False, "outcome": "LOSS", "pnl_r": -1.0,
             "pnl_usd": None, "shares": None, "mae_r": 1.0, "mfe_r": 0.1},
            {"id": 3, "symbol": "SPY", "date": "2026-06-30", "setup": "HOD_BREAK",
             "side": "LONG", "trigger_ts": "2026-06-30T10:30:00",
             "entry_ts": "2026-06-30T10:33:00", "entry_px": 600.0,
             "stop_px": 598.0, "target_px": 606.0,
             "exit_ts": "2026-06-30T15:54:00", "exit_px": 598.0,
             "exit_reason": "stop", "prob": 0.70, "threshold": 0.55,
             "taken": True, "outcome": "LOSS", "pnl_r": -1.0,
             "pnl_usd": -510.0, "shares": 250, "mae_r": 1.0, "mfe_r": 0.4},
        ],
    }

    def test_window_filter_actions_and_kpis(self):
        from engine import report
        rep = report.build_report(self.ART, date(2026, 5, 1), date(2026, 5, 31))
        assert rep["kpis"]["signals"] == 2          # SPY (June) excluded
        assert rep["kpis"]["taken"] == 1 and rep["kpis"]["filled"] == 1
        assert rep["kpis"]["win_rate"] == 100.0
        assert rep["kpis"]["net_pnl_usd"] == 500.0
        by_row = {r["symbol"]: r for r in rep["rows"]}
        assert by_row["NVDA"]["entry_action"] == "BUY"
        assert by_row["NVDA"]["exit_action"] == "SELL"
        assert by_row["TSLA"]["entry_action"] == "SELL SHORT"
        assert by_row["TSLA"]["exit_action"] == "COVER"
        assert by_row["NVDA"]["day"] == "Mon" and by_row["TSLA"]["day"] == "Tue"

    def test_files_written_and_note_for_precoverage(self, tmp_path):
        import csv as csv_mod
        from engine import report
        rep = report.build_report(self.ART, date(2026, 1, 1), date(2026, 6, 30))
        assert rep["note"] and "2026-05-04" in rep["note"]
        md, csvp = report.write_report(rep, tmp_path)
        text = md.read_text()
        assert "BUY → SELL" in text               # taken long shows its actions
        assert "10:36 @ 100.00" in text            # entry date+time+px present
        assert "| 2026-05-05 | Tue | TSLA" in text  # skipped short listed too
        with csvp.open() as fh:
            rows = list(csv_mod.DictReader(fh))
        assert len(rows) == 3
        assert rows[0]["entry_action"] == "BUY"
        assert rows[1]["taken"] == "False"
