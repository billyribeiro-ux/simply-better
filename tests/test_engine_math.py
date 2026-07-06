"""Unit tests for the engine's load-bearing math.

Covers the Part 4B required cases: outcome semantics, first-crossing
construction, geometry expectancy argmax, DSR monotonicity, the daily-context
no-leak shift, and attribution fold gating.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from engine import features, geometry, labeling, metrics
from engine.attribution import AdjustmentRule, Attribution, Condition
from engine.config import load_config
from engine.labeling import PathScan, outcome_for

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def _ts_list(n: int) -> list[datetime]:
    t0 = datetime(2025, 1, 6, 10, 0)
    return [t0 + timedelta(minutes=i) for i in range(n)]


def _scan_from_arrays(adv_cross: np.ndarray, fav_cross: np.ndarray,
                      eod_signed_atr: float, n_bars: int = 10) -> PathScan:
    ts = _ts_list(n_bars)
    return PathScan(
        event_id=0, entry_ts=ts[0], entry_px=100.0,
        adv_cross=adv_cross.astype(np.int32), fav_cross=fav_cross.astype(np.int32),
        eod_signed_atr=eod_signed_atr, eod_ts=ts[-1], eod_px=100.0, cross_ts=ts,
    )


def _scan_from_paths(grid: np.ndarray, fav: np.ndarray, adv: np.ndarray,
                     eod_signed_atr: float) -> PathScan:
    """Build a PathScan using the exact first-crossing construction from
    labeling.scan_paths (running max + searchsorted, never-crossed = -1)."""
    run_fav = np.maximum.accumulate(fav)
    run_adv = np.maximum.accumulate(adv)
    fav_cross = np.searchsorted(run_fav, grid, side="left")
    adv_cross = np.searchsorted(run_adv, grid, side="left")
    L = len(run_fav)
    fav_cross = np.where(fav_cross >= L, -1, fav_cross).astype(np.int32)
    adv_cross = np.where(adv_cross >= L, -1, adv_cross).astype(np.int32)
    ts = _ts_list(L)
    return PathScan(
        event_id=0, entry_ts=ts[0], entry_px=100.0,
        adv_cross=adv_cross, fav_cross=fav_cross,
        eod_signed_atr=eod_signed_atr, eod_ts=ts[-1], eod_px=100.0, cross_ts=ts,
    )


# ---------------------------------------------------------------------------
# 1. labeling.outcome_for
# ---------------------------------------------------------------------------
class TestOutcomeFor:
    GRID = np.array([0.5, 1.0, 1.5, 2.0])  # stop 1.0 -> idx 1, target 1.5 -> idx 2

    def test_target_first_is_win_with_t_over_s(self):
        scan = _scan_from_arrays(np.array([0, 7, -1, -1]), np.array([0, 1, 3, -1]), 1.8)
        win, pnl_r, reason, xi = outcome_for(scan, self.GRID, 1.0, 1.5)
        assert (win, reason, xi) == (1, "target", 3)
        assert pnl_r == pytest.approx(1.5 / 1.0)

    def test_stop_first_is_minus_one_r(self):
        scan = _scan_from_arrays(np.array([1, 2, -1, -1]), np.array([0, 3, 5, -1]), 0.4)
        win, pnl_r, reason, xi = outcome_for(scan, self.GRID, 1.0, 1.5)
        assert (win, reason, xi) == (0, "stop", 2)
        assert pnl_r == pytest.approx(-1.0)

    def test_same_bar_tie_resolves_to_loss(self):
        scan = _scan_from_arrays(np.array([1, 4, -1, -1]), np.array([0, 2, 4, -1]), 0.0)
        win, pnl_r, reason, _ = outcome_for(scan, self.GRID, 1.0, 1.5)
        assert (win, reason) == (0, "stop")
        assert pnl_r == pytest.approx(-1.0)

    def test_neither_crossed_is_eod(self):
        scan = _scan_from_arrays(
            np.array([0, -1, -1, -1]), np.array([0, -1, -1, -1]), 0.6, n_bars=10)
        win, pnl_r, reason, xi = outcome_for(scan, self.GRID, 1.0, 1.5)
        assert (win, reason, xi) == (1, "eod", 9)
        assert pnl_r == pytest.approx(0.6 / 1.0)

        scan = _scan_from_arrays(
            np.array([0, -1, -1, -1]), np.array([0, -1, -1, -1]), -0.3, n_bars=10)
        win, pnl_r, reason, xi = outcome_for(scan, self.GRID, 1.0, 1.5)
        assert (win, reason, xi) == (0, "eod", 9)
        assert pnl_r == pytest.approx(-0.3 / 1.0)


# ---------------------------------------------------------------------------
# 2. first-crossing semantics of the scan_paths construction
# ---------------------------------------------------------------------------
class TestFirstCrossing:
    def test_indices_match_hand_computation(self):
        grid = np.array([0.1, 0.5, 1.0, 2.0])
        fav = np.array([0.2, 0.5, 0.4, 1.1])          # per-bar excursion
        run_fav = np.maximum.accumulate(fav)          # [0.2, 0.5, 0.5, 1.1]
        cross = np.searchsorted(run_fav, grid, side="left")
        L = len(run_fav)
        cross = np.where(cross >= L, -1, cross)
        # 0.1 first reached at bar 0 (0.2 >= 0.1); 0.5 exactly at bar 1;
        # 1.0 at bar 3 (1.1); 2.0 never -> -1
        assert cross.tolist() == [0, 1, 3, -1]

    def test_exact_equality_counts_as_crossed(self):
        grid = np.array([0.5])
        run = np.maximum.accumulate(np.array([0.1, 0.5, 0.9]))
        idx = np.searchsorted(run, grid, side="left")
        assert idx.tolist() == [1]

    def test_never_crossed_is_minus_one(self):
        grid = np.array([3.0])
        run = np.maximum.accumulate(np.array([0.1, 0.2, 0.3]))
        cross = np.searchsorted(run, grid, side="left")
        cross = np.where(cross >= len(run), -1, cross)
        assert cross.tolist() == [-1]


# ---------------------------------------------------------------------------
# 3. geometry._search_cell picks the expectancy argmax
# ---------------------------------------------------------------------------
class TestSearchCell:
    def test_argmax_pair_selected(self, cfg):
        grid = cfg.scan_grid
        # scan 1: strong winner — fav ramps 0.1..2.0 over 20 bars, adverse
        # capped at 0.4; scan 2: fav caps at 1.0, adverse at 0.4; scan 3:
        # loser — adverse ramps to 1.0 in 5 bars, fav only arrives at bar 24.
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

        cell = geometry._search_cell(cfg, grid, scans, [0, 1, 2], "HOD_FADE", 1)

        # candidate sets from the observed excursion distributions:
        # max_adv = [0.4, 0.4, 1.0] -> quantiles (.55/.65/.75/.85) snapped ->
        # stops {0.5, 0.6, 0.7, 0.8}; max_fav = [2.0, 1.0, 1.0] -> targets
        # {1.0, 1.1, 1.3, 1.5}. Independent brute force over those pairs:
        paths = [(s1, 1.8), (s2, 0.9), (s3, -0.8)]
        favs = [np.maximum.accumulate(p) for p in (
            np.concatenate([grid[:20], np.full(10, 2.0)]),
            np.concatenate([grid[:10], np.full(20, 1.0)]),
            np.concatenate([np.full(24, 0.05), np.full(6, 1.0)]))]
        advs = [np.maximum.accumulate(p) for p in (
            np.concatenate([grid[:4], np.full(26, 0.4)]),
            np.concatenate([grid[:4], np.full(26, 0.4)]),
            np.concatenate([np.array([0.2, 0.4, 0.6, 0.8]), np.full(26, 1.0)]))]
        eods = [1.8, 0.9, -0.8]

        def first_cross(run: np.ndarray, level: float) -> int | None:
            hits = np.nonzero(run >= level)[0]
            return int(hits[0]) if len(hits) else None

        best_pair, best_exp, best_pwin = None, -np.inf, 0.0
        for s in (0.5, 0.6, 0.7, 0.8):
            for t in (1.0, 1.1, 1.3, 1.5):
                wins = stops = 0
                eod_sum = eod_n = 0
                for i in range(3):
                    f = first_cross(favs[i], t)
                    a = first_cross(advs[i], s)
                    if f is not None and (a is None or f < a):
                        wins += 1
                    elif a is not None:
                        stops += 1
                    else:
                        eod_sum += eods[i] / s
                        eod_n += 1
                p_win, p_stop, p_eod = wins / 3, stops / 3, eod_n / 3
                eod_r = eod_sum / eod_n if eod_n else 0.0
                exp = p_win * (t / s) - p_stop + p_eod * eod_r
                if exp > best_exp:
                    best_pair, best_exp, best_pwin = (s, t), exp, p_win

        assert best_pair == (0.5, 1.5)  # sanity: hand-computed argmax
        assert (cell.stop_atr, cell.target_atr) == best_pair
        assert cell.expectancy_r == pytest.approx(best_exp)
        assert cell.p_win == pytest.approx(best_pwin)
        assert cell.samples == 3


# ---------------------------------------------------------------------------
# 4. metrics.deflated_sharpe monotonicity in n_trials
# ---------------------------------------------------------------------------
class TestDeflatedSharpe:
    def test_dsr_never_increases_with_more_trials(self):
        rng = np.random.default_rng(11)
        rets = rng.normal(0.001, 0.01, 120)
        dsrs = [metrics.deflated_sharpe(rets, n) for n in (1, 2, 4, 16, 64, 256, 1024)]
        assert all(b <= a + 1e-12 for a, b in zip(dsrs, dsrs[1:]))
        assert dsrs[0] == pytest.approx(metrics.probabilistic_sharpe(rets))


# ---------------------------------------------------------------------------
# 5. features.daily_context no-leak shift
# ---------------------------------------------------------------------------
class TestDailyContextShift:
    @staticmethod
    def _daily(n: int, seed: int = 3) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        opens = closes * (1 + rng.normal(0, 0.004, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n)))
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
        return pl.DataFrame({
            "date": dates, "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": rng.uniform(1e6, 5e6, n),
        })

    def test_row_d_equals_day_d_minus_1(self):
        daily = self._daily(80)
        out = features.daily_context(daily)
        h = daily["high"].to_numpy()
        lo = daily["low"].to_numpy()
        c = daily["close"].to_numpy()

        prev_c = np.concatenate(([np.nan], c[:-1]))
        tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
        atr14 = features._roll_mean(tr, 14)

        ph = out["prior_high"].to_numpy()
        plo = out["prior_low"].to_numpy()
        pc = out["prior_close"].to_numpy()
        atr = out["atr"].to_numpy()
        for d in range(1, 80):
            assert ph[d] == pytest.approx(h[d - 1])
            assert plo[d] == pytest.approx(lo[d - 1])
            assert pc[d] == pytest.approx(c[d - 1])
            if np.isfinite(atr14[d - 1]):
                assert atr[d] == pytest.approx(atr14[d - 1])
            else:
                assert not np.isfinite(atr[d]) or atr[d] is None

    def test_same_day_data_cannot_leak_into_context(self):
        daily = self._daily(80)
        out = features.daily_context(daily)
        # blow up the LAST day's bar; its own context row must not move
        mutated = daily.with_columns(
            pl.when(pl.col("date") == daily["date"][-1])
            .then(pl.col("high") * 10).otherwise(pl.col("high")).alias("high"))
        out2 = features.daily_context(mutated)
        last, last2 = out.row(-1, named=True), out2.row(-1, named=True)
        for col in ("atr", "gk_vol_z", "prior_high", "prior_low", "prior_close"):
            assert last2[col] == pytest.approx(last[col])


# ---------------------------------------------------------------------------
# 6. attribution fold gating
# ---------------------------------------------------------------------------
class TestAttributionGating:
    @staticmethod
    def _attr(bump: float = 0.07) -> Attribution:
        rule = AdjustmentRule(
            conditions=[Condition("gap_pct", ">", 1.0)],
            loss_rate=0.7, support=40, bump=bump,
            learned_after_fold="2025-03",
        )
        return Attribution(rules=[rule])

    def test_rule_does_not_fire_on_its_own_fold(self):
        a = self._attr()
        row = {"gap_pct": 2.0}
        assert a.effective_threshold(row, 0.55, "2025-03") == pytest.approx(0.55)

    def test_rule_fires_on_strictly_later_fold(self):
        a = self._attr()
        row = {"gap_pct": 2.0}
        assert a.effective_threshold(row, 0.55, "2025-04") == pytest.approx(0.62)

    def test_rule_does_not_fire_when_conditions_miss(self):
        a = self._attr()
        row = {"gap_pct": 0.5}
        assert a.effective_threshold(row, 0.55, "2025-04") == pytest.approx(0.55)

    def test_threshold_capped_at_095(self):
        a = self._attr(bump=0.9)
        row = {"gap_pct": 2.0}
        assert a.effective_threshold(row, 0.55, "2025-04") == pytest.approx(0.95)
