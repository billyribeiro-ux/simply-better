"""Self-diagnosis: the machine measures WHY its market is (un)predictable.

Runs nightly inside `mie daily` and on demand via `mie diagnose`. Computes,
from the raw tape, the structural quantities that bound any model's possible
skill — so the engine itself reports how much predictability its universe
contains and where it lives, updated as new sessions arrive:

- signal-to-noise per decision horizon (drift vs noise);
- the ORACLE CEILING: accuracy of a trader who knows the day's final
  direction in advance — the hard upper bound on intraday direction calls;
- variance ratios (1.0 = random walk = nothing to learn from the path);
- HOD/LOD structure: how often the running extreme is broken again (false
  "the top is in" moments) and when final extremes form (the U-shape).

Output: dashboard/static/data/market_diagnosis.json (+ DuckDB-free, pure
tape computation; no model, no tuning, nothing to overfit).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config
from .data import Store

log = logging.getLogger("engine.diagnose")

HORIZON_MIN = 15          # matches the DL decision horizon


def run(cfg: Config) -> dict:
    store = Store(cfg)
    from datetime import date
    far, today = date(2000, 1, 1), date(2100, 1, 1)

    snr, oracle, vr2, vr6 = [], [], [], []
    fake_tops, hod_edge, lod_edge, nsess = [], 0, 0, 0

    for sym in cfg.universe:
        b1 = store.load_bars("1min", sym, far, today).sort("ts")
        b1 = b1.with_columns(pl.col("ts").dt.date().alias("dt"))
        for (day,), g in b1.group_by(["dt"], maintain_order=True):
            c = g["close"].to_numpy()
            h = g["high"].to_numpy()
            lo = g["low"].to_numpy()
            if len(c) < 300:
                continue
            nsess += 1
            r1 = np.diff(np.log(c))
            k = HORIZON_MIN
            nb = len(r1) // k
            rk = np.add.reduceat(r1, np.arange(0, nb * k, k))
            if rk.std() > 0:
                snr.append(abs(rk.mean()) / rk.std())
            ddir = np.sign(rk.sum())
            if ddir != 0:
                oracle.append(float((np.sign(rk) == ddir).mean()))
            if r1.std() > 0:
                vr2.append(float(np.var(np.add.reduceat(
                    r1, np.arange(0, (len(r1)//2)*2, 2))) / (2*np.var(r1))))
                vr6.append(float(np.var(np.add.reduceat(
                    r1, np.arange(0, (len(r1)//6)*6, 6))) / (6*np.var(r1))))
            # extremes structure (on 5-min-equivalent granularity: every 5th)
            run_hi = np.maximum.accumulate(h)
            fake_tops.append(int((np.diff(run_hi[30:]) > 0).sum()))
            edge = 30                                   # first/last 30 min
            if int(np.argmax(h)) < edge or int(np.argmax(h)) >= len(h)-edge:
                hod_edge += 1
            if int(np.argmin(lo)) < edge or int(np.argmin(lo)) >= len(lo)-edge:
                lod_edge += 1

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": nsess,
        "horizon_min": HORIZON_MIN,
        "snr_median": round(float(np.median(snr)), 4),
        "noise_to_signal": round(1.0 / float(np.median(snr)), 1),
        "oracle_ceiling_pct": round(100 * float(np.mean(oracle)), 2),
        "predictable_space_pts": round(100 * float(np.mean(oracle)) - 50.0, 2),
        "variance_ratio_2min": round(float(np.mean(vr2)), 4),
        "variance_ratio_6min": round(float(np.mean(vr6)), 4),
        "false_top_moments_per_day": round(float(np.mean(fake_tops)), 2),
        "hod_at_edges_pct": round(100 * hod_edge / max(nsess, 1), 1),
        "lod_at_edges_pct": round(100 * lod_edge / max(nsess, 1), 1),
        "reading": (
            "Oracle ceiling is the accuracy of perfect advance knowledge of "
            "the day's direction; the gap between 50% and it is ALL the "
            "predictability the tape contains at this horizon. Variance "
            "ratio 1.0 = random walk."),
    }
    log.info("diagnosis: SNR %.3f | oracle %.1f%% | VR6 %.3f | %d sessions",
             out["snr_median"], out["oracle_ceiling_pct"],
             out["variance_ratio_6min"], nsess)
    return out


def write(cfg: Config, diag: dict) -> Path:
    out = cfg.export_dir / "market_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(diag, fh, indent=2)
    return out
