"""MIE-DL trade layer: path-walk semantics, net-EV arithmetic vs costs.py,
blotter schema compatibility, and inference determinism (eval/live parity)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine import costs                        # noqa: E402
from engine.config import load_config           # noqa: E402
from engine.dl.dataset import SymbolCache       # noqa: E402
from engine.dl import signals as sig            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def _cache_from_bars(bars):
    """bars: list of (o,h,l,c) 1-min starting 10:00; single session."""
    d = date(2025, 3, 10)
    n = len(bars)
    ts = np.array([np.datetime64(datetime(2025, 3, 10, 10, 0)
                                 + timedelta(minutes=i), "us")
                   for i in range(n)])
    o, h, lo, c = (np.array([b[i] for b in bars], dtype=float)
                   for i in range(4))
    return SymbolCache(
        symbol="AAPL", ts1=ts,
        date1=np.array([np.datetime64(d, "D")] * n),
        open1=o, high1=h, low1=lo, close1=c,
        feat1=np.zeros((n, 9), np.float32), sigma1=np.full(n, 0.001),
        ts5=ts[:1], feat5=np.zeros((1, 6), np.float32),
        valid1=np.ones(n, bool))


class TestWalkPath:
    def test_target_hit_long(self):
        bars = [(100, 100.2, 99.9, 100.1)] * 3 + [(100.1, 101.5, 100.0, 101.2)] + \
               [(101.2, 101.3, 101.0, 101.1)] * 5
        cache = _cache_from_bars(bars)
        # decision bar 0; entry next open = bars[1].open = 100; unit = 1.0
        out = sig.walk_path(cache, 0, side=1.0, unit=1.0,
                            k_stop=1.0, k_target=1.0, time_exit_min=30)
        assert out.exit_reason == "target" and out.win == 1
        assert out.pnl_r_gross == pytest.approx(1.0)
        assert out.exit_px == pytest.approx(out.entry_px + 1.0)

    def test_same_bar_tie_is_stop(self):
        # one bar spans BOTH stop and target -> must resolve against the trade
        bars = [(100, 100.1, 99.9, 100.0)] * 2 + [(100.0, 101.5, 98.5, 100.0)] + \
               [(100, 100.1, 99.9, 100.0)] * 4
        cache = _cache_from_bars(bars)
        out = sig.walk_path(cache, 0, side=1.0, unit=1.0,
                            k_stop=1.0, k_target=1.0, time_exit_min=30)
        assert out.exit_reason == "stop" and out.win == 0
        assert out.pnl_r_gross == pytest.approx(-1.0)

    def test_time_exit(self):
        bars = [(100, 100.05, 99.95, 100.02)] * 40   # nothing ever hits
        cache = _cache_from_bars(bars)
        out = sig.walk_path(cache, 0, side=1.0, unit=1.0,
                            k_stop=1.0, k_target=1.0, time_exit_min=10)
        assert out.exit_reason == "time"
        # exited at close of the last bar inside the 10-min window
        assert out.exit_idx <= 11

    def test_short_side_mirrors(self):
        bars = [(100, 100.1, 99.9, 100.0)] * 3 + [(100.0, 100.2, 98.4, 98.6)] + \
               [(98.6, 98.7, 98.5, 98.6)] * 4
        cache = _cache_from_bars(bars)
        out = sig.walk_path(cache, 0, side=-1.0, unit=1.0,
                            k_stop=1.0, k_target=1.0, time_exit_min=30)
        assert out.exit_reason == "target" and out.win == 1


class TestNetEv:
    def test_net_pnl_matches_costs_module(self, cfg):
        # hand-check: net = gross - cost_r
        sym, px, unit, k_stop = "AAPL", 200.0, 0.8, 1.0
        comm = float(cfg.execution["commission_per_share"])
        expected_cost = costs.cost_r(costs.slip_frac(cfg, sym), comm, px,
                                     k_stop, unit)
        net = sig.net_pnl_r(cfg, sym, px, unit, 1.0, k_stop)
        assert net == pytest.approx(1.0 - expected_cost, abs=1e-12)


class TestSchema:
    def test_blotter_keys_superset_of_engine_schema(self):
        needed = {"id", "symbol", "date", "setup", "side", "trigger_ts",
                  "entry_ts", "entry_px", "stop_px", "target_px", "exit_ts",
                  "exit_px", "exit_reason", "prob", "threshold", "taken",
                  "outcome", "pnl_r", "pnl_usd", "shares", "mae_r", "mfe_r",
                  "net_ev_r"}
        import inspect
        src = inspect.getsource(sig.build_trade_rows)
        for key in needed:
            assert f'"{key}"' in src, f"blotter row missing key {key}"


class TestInferenceParity:
    def test_same_inputs_same_scores(self, cfg):
        from engine.dl.model import CausalTCN
        from engine.dl.train import set_determinism
        set_determinism(7)
        m = CausalTCN(); m.eval()
        x1 = torch.randn(16, 9, 64); x5 = torch.randn(16, 6, 48)
        sym = torch.randint(0, 8, (16,)); st = torch.randn(16, 2)
        with torch.no_grad():
            s1 = CausalTCN.score(m(x1, x5, sym, st)[0])
            s2 = CausalTCN.score(m(x1, x5, sym, st)[0])
        assert torch.equal(s1, s2)
