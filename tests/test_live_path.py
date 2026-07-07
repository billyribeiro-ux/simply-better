"""Train/serve parity and live-path tests.

Why not exact probability parity against a walk-forward fold? The fold's
model deliberately excludes the embargo gap (last 5 days before the fold
month) — a production model trained through the end of the prior month
legitimately includes them, so its probabilities differ BY DESIGN. What must
be identical, and is asserted here:

- the trigger set and features for a session (same detector on same bars),
- the geometry (production trained to the fold's train boundary reproduces
  the fold's learned cells exactly),
- confirmation entries and stop/target prices for the backtest's decided
  signals,
- the scoring plumbing (live probs == model.score on the same matrix; the
  fitting function itself is shared code with walk_forward).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from engine import geometry as geometry_mod
from engine import labeling, learn, live, model, production, setups
from engine.attribution import Attribution
from engine.config import load_config
from engine.data import Store
from engine.features import FEATURES, daily_context
from engine.geometry import GeoCell, Geometry
from engine.production import ProductionBundle

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tmp_cfg(tmp_path: Path, universe: list[str]):
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["universe"] = universe
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(p)


def _gen_symbol(sym, start, n_days, px0, rng):
    """Synthetic session generator (same construction as the smoke test)."""
    daily_rows, b5, b1 = [], [], []
    px, d, made = px0, start, 0
    while made < n_days:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
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
            b1.append((t0 + timedelta(minutes=i), float(opens[i]),
                       float(max(opens[i], highs[i])), float(min(opens[i], lows[i])),
                       float(closes[i]), float(vols[i]), sym))
        for i in range(0, mins, 5):
            sl = slice(i, i + 5)
            b5.append((t0 + timedelta(minutes=i), float(opens[i]), float(highs[sl].max()),
                       float(lows[sl].min()), float(closes[i + 4]), float(vols[sl].sum()), sym))
        daily_rows.append((d, float(o), float(highs.max()), float(lows.min()),
                           float(closes[-1]), float(vols.sum())))
        px = closes[-1]
        made += 1
        d += timedelta(days=1)
    schema5 = ["ts", "open", "high", "low", "close", "volume", "symbol"]
    return (pl.DataFrame(daily_rows, schema=["date", "open", "high", "low", "close", "volume"], orient="row"),
            pl.DataFrame(b5, schema=schema5, orient="row"),
            pl.DataFrame(b1, schema=schema5, orient="row"))


@pytest.fixture(scope="module")
def research_env(tmp_path_factory):
    """Synthetic 10-month research environment + full backtest artifacts."""
    tmp = tmp_path_factory.mktemp("live_parity")
    cfg = _tmp_cfg(tmp, ["AAPL", "TSLA"])
    store = Store(cfg)
    rng = np.random.default_rng(7)
    start = date(2024, 1, 2)
    for sym, px0 in [("AAPL", 190.0), ("TSLA", 240.0)]:
        daily, b5, b1 = _gen_symbol(sym, start, 210, px0, rng)
        store.save_daily(sym, daily)
        store.save_bars("5min", sym, b5)
        store.save_bars("1min", sym, b1)
    art = learn.run_pipeline(cfg, start, date(2024, 11, 30))
    return cfg, start, art


@pytest.fixture(scope="module")
def fold_bundle(research_env):
    """Production bundle trained through the last fold's train boundary."""
    cfg, start, art = research_env
    last_fold = art["diagnostics"]["folds"][-1]["fold"]         # "YYYY-MM"
    y, m = map(int, last_fold.split("-"))
    cutoff = date(y, m, 1) - timedelta(days=1)                  # end of prior month
    bundle = production.train_production(cfg, start, cutoff)
    return cfg, art, last_fold, bundle


class TestFoldParity:
    def test_geometry_matches_last_fold(self, fold_bundle):
        _, art, _, bundle = fold_bundle
        # the last fold's geometry (geo_final in artifacts) was learned on all
        # events before the fold month — exactly the production training set
        assert bundle.geometry.rows() == art["geometry"]

    def test_replay_reproduces_backtest_triggers_and_prices(self, fold_bundle):
        cfg, art, last_fold, bundle = fold_bundle
        fold_signals = [s for s in art["signals"] if s["date"][:7] == last_fold]
        assert fold_signals, "no decided signals in the last fold"
        day = date.fromisoformat(fold_signals[0]["date"])
        expected = {(s["symbol"], s["trigger_ts"]): s
                    for s in fold_signals if s["date"] == day.isoformat()}

        report = live.scan_live(cfg, bundle, day,
                                now_et=datetime.combine(day, time(16, 0)))
        got = {(s["symbol"], s["trigger_ts"]): s for s in report["signals"]}

        # every decided backtest signal must reappear in the live replay
        assert set(expected).issubset(set(got))
        for key, exp in expected.items():
            g = got[key]
            assert g["status"] == "confirmed"
            assert g["entry_px"] == pytest.approx(exp["entry_px"], abs=0.011)
            # identical geometry (asserted above) + identical entry px
            # => identical stop and target prices
            assert g["stop_px"] == pytest.approx(exp["stop_px"], abs=0.011)
            assert g["target_px"] == pytest.approx(exp["target_px"], abs=0.011)
            # probabilities are NOT compared: the fold's model excludes the
            # embargo gap by design; the production model includes it.


class TestScoringPlumbingParity:
    def test_live_probs_threshold_and_sizing_match_direct_composition(self, fold_bundle):
        cfg, art, last_fold, bundle = fold_bundle
        fold_signals = [s for s in art["signals"] if s["date"][:7] == last_fold]
        day = date.fromisoformat(fold_signals[0]["date"])
        now = datetime.combine(day, time(16, 0))
        report = live.scan_live(cfg, bundle, day, now_et=now)
        assert report["signals"]

        # independent recomposition from the same store
        store = Store(cfg)
        events = []
        for sym in bundle.universe:
            ctx = daily_context(store.load_daily(sym))
            b5 = store.load_bars("5min", sym, day - timedelta(days=14), day)
            b5 = b5.filter(pl.col("ts") + pl.duration(minutes=5) <= now)
            events.extend(setups.detect(cfg, sym, b5, ctx))
        today = sorted((e for e in events if e["session_date"] == day),
                       key=lambda e: e["trigger_ts"])
        assert len(today) == len(report["signals"])

        X = np.array([[float(e[f]) for f in FEATURES] for e in today])
        probs = model.score(bundle.fm, X)
        attribution = Attribution(rules=list(bundle.rules))
        ex = cfg.execution
        risk_usd = float(ex["starting_equity"]) * float(ex["risk_per_trade_pct"]) / 100.0

        gate_max = float(cfg.setups.get("day_type", {}).get("eff_gate_max", 1e9))
        for ev, prob, sig in zip(today, probs, report["signals"]):
            assert sig["symbol"] == ev["symbol"]
            assert sig["prob"] == pytest.approx(float(prob), abs=5e-4)
            eff = attribution.effective_threshold(ev, bundle.fm.threshold, "9999-12")
            assert sig["threshold"] == pytest.approx(eff, abs=5e-4)
            stop_atr, target_atr, tradable = bundle.geometry.lookup(ev)
            assert sig["tradable"] == tradable
            veto = float(ev.get("dt_eff_h1_aligned", 0.0)) > gate_max
            assert sig["trend_veto"] == veto
            assert sig["taken"] == (tradable and float(prob) >= eff and not veto)
            if sig["shares"] and stop_atr * ev["atr"] > 0:
                assert sig["shares"] == int(risk_usd // (stop_atr * ev["atr"]))


class TestBundlePersistence:
    def test_round_trip_preserves_probabilities(self, fold_bundle, tmp_path):
        cfg, _, _, bundle = fold_bundle
        production.save_bundle(cfg, bundle)
        loaded = production.load_bundle(cfg)
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (32, len(FEATURES)))
        assert np.allclose(model.score(bundle.fm, X), model.score(loaded.fm, X))
        assert loaded.geometry.rows() == bundle.geometry.rows()
        assert loaded.fm.threshold == bundle.fm.threshold

    def test_loader_rejects_feature_mismatch(self, fold_bundle):
        import pickle
        cfg, _, _, bundle = fold_bundle
        p = production.bundle_path(cfg)
        stale = pickle.loads(p.read_bytes())
        stale.features = stale.features[:-1]        # simulate code drift
        p.write_bytes(pickle.dumps(stale))
        with pytest.raises(RuntimeError, match="feature set"):
            production.load_bundle(cfg)
        production.save_bundle(cfg, bundle)         # restore for other tests


# ---------------------------------------------------------------------------
# live status semantics on a hand-crafted session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crafted_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("live_status")
    cfg = _tmp_cfg(tmp, ["TEST"])
    store = Store(cfg)
    day = date(2025, 3, 10)

    # 30 daily rows ending on the session date (ATR ~ 2.0)
    rows = []
    for i in range(30):
        d = day - timedelta(days=29 - i)
        c = 100.0 + 0.05 * i
        rows.append((d, c + 0.2, c + 1.2, c - 0.8, c, 5e6))
    store.save_daily("TEST", pl.DataFrame(
        rows, schema=["date", "open", "high", "low", "close", "volume"], orient="row"))

    # 5-min session: HOD_FADE trigger at 10:00 (trigger_low 100.9),
    # LOD_RECLAIM trigger at 10:05 (trigger_high 99.7)
    bars = [
        (100.00, 100.20, 99.80, 100.10, 1000.0),
        (100.10, 100.30, 99.90, 100.00, 1200.0),
        (100.00, 100.10, 99.70, 99.90, 900.0),
        (99.99, 100.02, 100.00, 100.01, 50000.0),
        (100.01, 100.25, 99.85, 100.05, 1100.0),
        (100.05, 100.30, 99.90, 100.15, 1000.0),
        (101.20, 102.00, 100.90, 101.00, 8000.0),   # 10:00 HOD FADE
        (99.30, 99.70, 99.00, 99.60, 800.0),        # 10:05 LOD RECLAIM
        (99.60, 99.80, 99.40, 99.70, 900.0),
        (99.70, 99.90, 99.50, 99.80, 900.0),
    ]
    bars[3] = (99.99, 100.02, 100.00, 100.01, 50000.0)
    t0 = datetime(2025, 3, 10, 9, 30)
    b5 = pl.DataFrame(
        [(t0 + timedelta(minutes=5 * i), o, h, lo, c, v, "TEST")
         for i, (o, h, lo, c, v) in enumerate(bars)],
        schema=["ts", "open", "high", "low", "close", "volume", "symbol"], orient="row")
    store.save_bars("5min", "TEST", b5)

    # 1-min bars 10:05..10:30, all trading ABOVE 101.5:
    #  - HOD break level is 100.89 -> never broken -> expires
    #  - LOD break level is 99.71  -> broken instantly -> confirms
    t1 = datetime(2025, 3, 10, 10, 5)
    b1 = pl.DataFrame(
        [(t1 + timedelta(minutes=i), 101.60, 102.00, 101.50, 101.80, 500.0, "TEST")
         for i in range(26)],
        schema=["ts", "open", "high", "low", "close", "volume", "symbol"], orient="row")
    store.save_bars("1min", "TEST", b1)

    # minimal production bundle: quick model + always-tradable atr geometry
    rng = np.random.default_rng(5)
    lab_rows = []
    for m in range(8):
        for k in range(40):
            d = date(2025, 1 + m, 1) + timedelta(days=int(k * 27 / 40))
            row = {"session_date": d,
                   "entry_ts": datetime(d.year, d.month, d.day, 10, 0) + timedelta(minutes=k),
                   "outcome": int(rng.random() < 0.5),
                   "pnl_r": float(rng.normal(0, 1))}
            for f in FEATURES:
                row[f] = float(rng.normal(0, 1))
            lab_rows.append(row)
    fm = model.fit_scored_model(cfg, pl.DataFrame(lab_rows).sort("entry_ts"))
    fb = GeoCell("*", -1, 0.5, 0.75, 0.5, 0.1, 50)
    geo = Geometry({}, (-0.4, 0.4), fb, grid=cfg.scan_grid, min_room_atr=0.3)
    bundle = ProductionBundle(
        fm=fm, geometry=geo, rules=[], trained_from=date(2025, 1, 1),
        trained_through=date(2025, 3, 9), universe=["TEST"],
        features=list(FEATURES), created_at="test", n_events=0, n_labeled=320)
    return cfg, bundle, day


class TestLiveStatuses:
    @staticmethod
    def _by_setup(report):
        return {s["setup"]: s for s in report["signals"]}

    def test_awaiting_before_confirmation_window_ends(self, crafted_env):
        cfg, bundle, day = crafted_env
        # the detector needs 8 completed 5-min bars, so the 10:00 trigger
        # surfaces at 10:10 (bounded detection latency, never lookahead).
        # At 10:12 only ~7 one-minute bars exist and none broke 100.89.
        report = live.scan_live(cfg, bundle, day,
                                now_et=datetime(2025, 3, 10, 10, 12))
        sig = self._by_setup(report)
        assert sig["HOD_FADE"]["status"] == "awaiting"
        assert sig["HOD_FADE"]["entry_px"] is None
        assert sig["HOD_FADE"]["entry_trigger_px"] == pytest.approx(100.89)

    def test_confirmed_and_expired_end_of_day(self, crafted_env):
        cfg, bundle, day = crafted_env
        report = live.scan_live(cfg, bundle, day,
                                now_et=datetime.combine(day, time(16, 0)))
        sig = self._by_setup(report)
        assert set(sig) == {"HOD_FADE", "LOD_RECLAIM"}
        assert sig["HOD_FADE"]["status"] == "expired"       # never traded below 100.89
        lod = sig["LOD_RECLAIM"]
        assert lod["status"] == "confirmed"                 # first bar high >= 99.71
        # stop-market fill semantics: max(open, break level)
        assert lod["entry_px"] == pytest.approx(101.60)
        # stop/target derived from actual entry via the label() formula
        atr = 2.0
        assert lod["stop_px"] == pytest.approx(101.60 - 0.5 * atr, abs=0.05)
        assert lod["target_px"] == pytest.approx(101.60 + 0.75 * atr, abs=0.05)

    def test_in_progress_bar_is_invisible(self, crafted_env):
        cfg, bundle, day = crafted_env
        # 10:03 -> the 10:00 trigger bar has not closed; nothing may be seen
        report = live.scan_live(cfg, bundle, day,
                                now_et=datetime(2025, 3, 10, 10, 3))
        assert report["signals"] == []
        assert report.get("note")
        # 10:07 -> some 1-min bars exist but the day still has < 8 completed
        # 5-min bars, so the 10:00 trigger is not yet visible
        report = live.scan_live(cfg, bundle, day,
                                now_et=datetime(2025, 3, 10, 10, 7))
        assert report["signals"] == []
        assert "no setup triggers" in report.get("note", "")
