"""Iteration-two tests: anchors, anchor-mode geometry, calibration hygiene."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from engine import geometry, model
from engine.anchors import PriceHistogram, anchor_room_atr
from engine.config import load_config
from engine.features import FEATURES
from engine.geometry import GeoCell, Geometry, _snap_level
from engine.labeling import PathScan
from engine.setups import detect

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# 1. PriceHistogram
# ---------------------------------------------------------------------------
class TestPriceHistogram:
    def test_volume_concentration_sets_poc(self):
        h = PriceHistogram(atr=5.0)           # bin_w = 0.1
        assert h.bin_w == pytest.approx(0.1)
        h.add_bar(high=100.05, low=100.00, close=100.02, volume=1000.0)
        h.add_bar(high=101.05, low=101.00, close=101.01, volume=10.0)
        # bin floor(100.0/0.1)=1000 -> center (1000+0.5)*0.1
        assert h.poc() == pytest.approx(100.05)

    def test_empty_returns_none(self):
        h = PriceHistogram(atr=1.0)
        assert h.poc() is None
        assert h.is_empty()

    def test_uniform_spread_conserves_volume(self):
        h = PriceHistogram(atr=2.0)           # bin_w = 0.04
        h.add_bar(high=100.55, low=100.00, close=100.30, volume=999.0)
        assert sum(h._vol.values()) == pytest.approx(999.0, abs=1e-9)

    def test_zero_volume_ignored(self):
        h = PriceHistogram(atr=2.0)
        h.add_bar(high=100.5, low=100.0, close=100.2, volume=0.0)
        assert h.is_empty()


# ---------------------------------------------------------------------------
# 2. anchor_room_atr sign cases
# ---------------------------------------------------------------------------
class TestAnchorRoom:
    def test_anchor_in_profit_direction_is_positive(self):
        # short from 100, anchor below at 99, atr 2 -> +0.5 ATR of runway
        assert anchor_room_atr(99.0, 100.0, -1.0, 2.0) == pytest.approx(0.5)
        # long from 100, anchor above at 101 -> +0.5
        assert anchor_room_atr(101.0, 100.0, 1.0, 2.0) == pytest.approx(0.5)

    def test_anchor_behind_is_negative(self):
        assert anchor_room_atr(101.0, 100.0, -1.0, 2.0) == pytest.approx(-0.5)
        assert anchor_room_atr(99.0, 100.0, 1.0, 2.0) == pytest.approx(-0.5)

    def test_none_anchor_or_bad_atr_is_zero(self):
        assert anchor_room_atr(None, 100.0, 1.0, 2.0) == 0.0
        assert anchor_room_atr(101.0, 100.0, 1.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 3. developing POC has no lookahead (through setups.detect)
# ---------------------------------------------------------------------------
class TestDevPocNoLookahead:
    @staticmethod
    def _day(cfg):
        """One 5-min session: HOD_FADE trigger at bar 6 carrying huge volume,
        LOD_RECLAIM trigger at bar 7. Bar 6's volume must NOT be inside bar
        6's own dev POC, and MUST be inside bar 7's."""
        day = date(2025, 3, 10)
        t0 = datetime(2025, 3, 10, 9, 30)
        #        open    high    low     close   volume
        bars = [
            (100.00, 100.20, 99.80, 100.10, 1000.0),   # 09:30
            (100.10, 100.30, 99.90, 100.00, 1200.0),   # 09:35
            (100.00, 100.10, 99.70, 99.90, 900.0),     # 09:40
            (99.90, 100.02, 100.00 - 0.00, 100.01, 50000.0),  # 09:45 tight, dominant
            (100.01, 100.25, 99.85, 100.05, 1100.0),   # 09:50
            (100.05, 100.30, 99.90, 100.15, 1000.0),   # 09:55
            (101.20, 102.00, 100.90, 101.00, 1e9),     # 10:00 HOD FADE + huge vol
            (99.30, 99.70, 99.00, 99.60, 800.0),       # 10:05 LOD RECLAIM
            (99.60, 99.80, 99.40, 99.70, 900.0),       # 10:10
            (99.70, 99.90, 99.50, 99.80, 900.0),       # 10:15
        ]
        # fix bar 3 low to a sane value (tight dominant bin at 100.00-100.02)
        bars[3] = (99.99, 100.02, 100.00, 100.01, 50000.0)
        rows = [
            (t0 + timedelta(minutes=5 * i), o, h, lo, c, v, "TEST")
            for i, (o, h, lo, c, v) in enumerate(bars)
        ]
        bars5 = pl.DataFrame(
            rows, schema=["ts", "open", "high", "low", "close", "volume", "symbol"],
            orient="row")
        ctx = pl.DataFrame({
            "date": [day], "atr": [2.0], "gk_vol_z": [0.0],
            "prior_high": [100.5], "prior_low": [99.5], "prior_close": [100.0],
        })
        return bars5, ctx, bars

    def test_bar_i_volume_excluded_then_included(self, cfg):
        bars5, ctx, bars = self._day(cfg)
        events = detect(cfg, "TEST", bars5, ctx)
        by_setup = {e["setup"]: e for e in events}
        assert "HOD_FADE" in by_setup and "LOD_RECLAIM" in by_setup
        ev_i = by_setup["HOD_FADE"]        # bar 6
        ev_j = by_setup["LOD_RECLAIM"]     # bar 7
        assert ev_i["trigger_ts"].minute == 0 and ev_i["trigger_ts"].hour == 10
        assert ev_j["trigger_ts"].minute == 5

        # independently rebuild the histograms with the same implementation
        before = PriceHistogram(2.0)
        for o, h, lo, c, v in bars[:6]:
            before.add_bar(h, lo, c, v)
        after = PriceHistogram(2.0)
        for o, h, lo, c, v in bars[:7]:
            after.add_bar(h, lo, c, v)

        assert ev_i["anchor_px"]["dev_poc"] == pytest.approx(before.poc())
        assert ev_j["anchor_px"]["dev_poc"] == pytest.approx(after.poc())
        # and the huge bar really moved it: bar-6 volume lives at 100.9-102.0
        assert ev_i["anchor_px"]["dev_poc"] < 100.5
        assert ev_j["anchor_px"]["dev_poc"] >= 100.9


# ---------------------------------------------------------------------------
# 4. anchor-mode _search_cell: throughput-adjusted argmax, no-trade handling
# ---------------------------------------------------------------------------
def _scan_from_paths(grid, fav, adv, eod):
    run_fav = np.maximum.accumulate(fav)
    run_adv = np.maximum.accumulate(adv)
    fav_cross = np.searchsorted(run_fav, grid, side="left")
    adv_cross = np.searchsorted(run_adv, grid, side="left")
    L = len(run_fav)
    fav_cross = np.where(fav_cross >= L, -1, fav_cross).astype(np.int32)
    adv_cross = np.where(adv_cross >= L, -1, adv_cross).astype(np.int32)
    t0 = datetime(2025, 1, 6, 10, 0)
    ts = [t0 + timedelta(minutes=i) for i in range(L)]
    return PathScan(0, ts[0], 100.0, adv_cross, fav_cross, eod, ts[-1], 100.0, ts)


class TestAnchorSearchCell:
    def _fixtures(self, cfg):
        grid = cfg.scan_grid
        s1 = _scan_from_paths(
            grid, np.concatenate([grid[:20], np.full(10, 2.0)]),
            np.concatenate([grid[:4], np.full(26, 0.4)]), 1.8)
        s2 = _scan_from_paths(
            grid, np.concatenate([grid[:10], np.full(20, 1.0)]),
            np.concatenate([grid[:4], np.full(26, 0.4)]), 0.9)
        s3 = _scan_from_paths(
            grid, np.concatenate([np.full(24, 0.05), np.full(6, 1.0)]),
            np.concatenate([np.array([0.2, 0.4, 0.6, 0.8]), np.full(26, 1.0)]), -0.8)
        scans = {0: s1, 1: s2, 2: s3}
        # only the vwap anchor has room; event 2 sits below min_room_atr (0.3)
        events = [
            {"room_vwap_atr": 2.0},
            {"room_vwap_atr": 1.0},
            {"room_vwap_atr": 0.2},
        ]
        return grid, scans, events

    def test_throughput_adjusted_argmax(self, cfg):
        grid, scans, events = self._fixtures(cfg)
        cell = geometry._search_cell(cfg, grid, scans, [0, 1, 2],
                                     "HOD_FADE", 1, events=events)
        # hand computation: vwap/frac=1.0/stop=0.5 trades events 0,1 (both
        # reach their own anchor targets 2.0 and 1.0 before any 0.5 stop):
        # E[R|traded] = (2.0/0.5 + 1.0/0.5)/2 = 3.0; throughput 2/3 -> 2.0,
        # beating atr-mode's best 1.2667 from the same scans.
        assert cell.anchor == "vwap"
        assert cell.frac == pytest.approx(1.0)
        assert cell.stop_atr == pytest.approx(0.5)
        assert cell.expectancy_r == pytest.approx(3.0 * 2 / 3)
        assert cell.p_win == pytest.approx(1.0)
        assert cell.samples == 3

    def test_sub_min_room_is_no_trade_not_loss(self, cfg):
        grid, scans, events = self._fixtures(cfg)
        cell = geometry._search_cell(cfg, grid, scans, [0, 1, 2],
                                     "HOD_FADE", 1, events=events)
        # event 2 is a certain loser (fast adverse). If it were counted as a
        # loss instead of no-trade, the winning score would be
        # (2/0.5 + 1/0.5 - 1)/3 = 1.667, and p_win 2/3 — not what we assert:
        assert cell.expectancy_r == pytest.approx(2.0)
        assert cell.p_win == pytest.approx(1.0)

    def test_events_none_reproduces_atr_mode(self, cfg):
        grid, scans, _ = self._fixtures(cfg)
        cell = geometry._search_cell(cfg, grid, scans, [0, 1, 2], "HOD_FADE", 1)
        assert cell.anchor == "atr"
        assert (cell.stop_atr, cell.target_atr) == (0.5, 1.5)
        assert cell.expectancy_r == pytest.approx(1.0 - 1 / 3 + 0.6, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Geometry.lookup for anchor cells
# ---------------------------------------------------------------------------
class TestGeometryLookup:
    def _geo(self, cfg, frac=0.618):
        cell = GeoCell("HOD_FADE", 1, 0.5, 1.0, 0.6, 1.0, 100,
                       anchor="vwap", frac=frac)
        fb = GeoCell("*", -1, 0.5, 0.75, 0.0, 0.0, 0)
        return Geometry({("HOD_FADE", 1): cell}, (-0.4, 0.4), fb,
                        grid=cfg.scan_grid, min_room_atr=0.3)

    def test_tradable_flips_exactly_at_min_room(self, cfg):
        geo = self._geo(cfg)
        ev = {"setup": "HOD_FADE", "gk_vol_z": 0.0, "room_vwap_atr": 0.3}
        assert geo.lookup(ev)[2] is True
        ev["room_vwap_atr"] = 0.2999999
        assert geo.lookup(ev)[2] is False

    def test_target_clamps_to_grid_bounds(self, cfg):
        geo = self._geo(cfg, frac=1.0)
        ev = {"setup": "HOD_FADE", "gk_vol_z": 0.0, "room_vwap_atr": 50.0}
        s, t, ok = geo.lookup(ev)
        assert ok and t == pytest.approx(3.0)          # grid max
        ev["room_vwap_atr"] = 0.30
        s, t, ok = geo.lookup(ev)
        assert ok and 0.1 <= t <= 3.0
        assert t == pytest.approx(_snap_level(0.30, cfg.scan_grid))

    def test_atr_cell_always_tradable(self, cfg):
        fb = GeoCell("*", -1, 0.5, 0.75, 0.0, 0.0, 0)
        geo = Geometry({}, (-0.4, 0.4), fb, grid=cfg.scan_grid, min_room_atr=0.3)
        ev = {"setup": "LOD_RECLAIM", "gk_vol_z": 0.0}
        assert geo.lookup(ev) == (0.5, 0.75, True)


# ---------------------------------------------------------------------------
# 6. calibration hygiene: isotonic fit sees ONLY the train tail
# ---------------------------------------------------------------------------
class TestCalibrationHygiene:
    @staticmethod
    def _labeled(n_months=8, per_month=40, seed=5) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for m in range(n_months):
            for k in range(per_month):
                d = date(2025, 1 + m, 1) + timedelta(days=int(k * 27 / per_month))
                ts = datetime(d.year, d.month, d.day, 10, 0) + timedelta(minutes=k)
                row = {
                    "session_date": d, "entry_ts": ts,
                    "outcome": int(rng.random() < 0.5),
                    "pnl_r": float(rng.normal(0, 1)),
                }
                for f in FEATURES:
                    row[f] = float(rng.normal(0, 1))
                rows.append(row)
        return pl.DataFrame(rows)

    def test_isotonic_fit_only_on_train_oof(self, cfg, monkeypatch):
        """The calibrator may only ever see TRAIN-window out-of-fold
        probabilities and outcomes — never test-fold labels, and never the
        final model's in-sample (memorized) probabilities."""
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        class SpyIso:
            def __init__(self, out_of_bounds=None):
                self._x = None

            def fit(self, x, y):
                calls.append((np.asarray(x).copy(), np.asarray(y).copy()))
                return self

            def predict(self, x):
                return np.asarray(x, dtype=float)   # identity transform

        monkeypatch.setattr(model, "IsotonicRegression", SpyIso)
        assert bool(cfg.model.get("calibrate")) is True

        labeled = self._labeled()
        folds = model.walk_forward(cfg, labeled)
        assert folds, "walk-forward produced no folds"
        assert len(calls) == len(folds)

        embargo = timedelta(days=int(cfg.model["walk_forward"]["embargo_days"]))
        K = int(cfg.model["oof_folds"])
        df = labeled.sort("entry_ts").with_columns(
            pl.col("session_date").dt.strftime("%Y-%m").alias("fold_month"))
        for fr, (x_fit, y_fit) in zip(folds, calls):
            month_start = df.filter(pl.col("fold_month") == fr.fold)["session_date"].min()
            train = df.filter(
                pl.col("session_date") < (month_start - embargo)).sort("entry_ts")
            # OOF coverage = blocks 1..K-1 (block 0 has no prior data)
            first_covered = int(round(train.height / K))
            expected = train["outcome"].to_numpy()[first_covered:]
            assert len(y_fit) == len(expected)      # exactly the OOF rows
            assert np.array_equal(y_fit, expected)  # train outcomes only
            assert len(y_fit) < train.height        # never the full train
            # the final model's in-sample probs would be ~0/1; OOF must not be
            assert 0.02 < float(np.mean((x_fit > 0.02) & (x_fit < 0.98)))
