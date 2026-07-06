"""Learned trade geometry.

For each (setup, vol_regime) cell, search stop/target candidates drawn from
the empirical MAE/MFE distributions of the TRAINING events only, and pick the
pair that maximizes expectancy in R:

    E = p_win * (t/s)  +  p_stop * (-1)  +  p_eod * E[eod_r | eod]

Because ``PathScan`` already holds first-crossing indices for the whole grid,
every candidate pair is evaluated with pure array comparisons — the exact
same path data, zero re-simulation, zero lookahead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config
from .labeling import PathScan

log = logging.getLogger("engine.geometry")


@dataclass(frozen=True)
class GeoCell:
    setup: str
    regime: int
    stop_atr: float
    target_atr: float
    p_win: float
    expectancy_r: float
    samples: int


class Geometry:
    def __init__(self, cells: dict[tuple[str, int], GeoCell],
                 regime_bounds: tuple[float, float], fallback: GeoCell) -> None:
        self.cells = cells
        self.regime_bounds = regime_bounds
        self.fallback = fallback

    def regime_of(self, gk_vol_z: float) -> int:
        lo, hi = self.regime_bounds
        if gk_vol_z <= lo:
            return 0
        if gk_vol_z <= hi:
            return 1
        return 2

    def lookup(self, ev: dict) -> tuple[float, float]:
        cell = self.cells.get((ev["setup"], self.regime_of(float(ev["gk_vol_z"]))),
                              self.fallback)
        return cell.stop_atr, cell.target_atr

    def rows(self) -> list[dict]:
        return [
            {"setup": c.setup, "regime": c.regime, "stop_atr": c.stop_atr,
             "target_atr": c.target_atr, "p_win": round(c.p_win, 4),
             "expectancy_r": round(c.expectancy_r, 4), "samples": c.samples}
            for c in sorted(self.cells.values(), key=lambda x: (x.setup, x.regime))
        ]


def learn(cfg: Config, events: list[dict], scans: dict[int, PathScan],
          train_ids: list[int]) -> Geometry:
    grid = cfg.scan_grid
    gcfg = cfg.geometry
    min_samples = int(gcfg["min_samples"])

    train = [eid for eid in train_ids if eid in scans]
    if not train:
        fb = GeoCell("*", -1, 0.5, 0.75, 0.0, 0.0, 0)
        return Geometry({}, (-0.43, 0.43), fb)

    zs = np.array([float(events[eid]["gk_vol_z"]) for eid in train])
    bounds = (float(np.quantile(zs, 1 / 3)), float(np.quantile(zs, 2 / 3)))

    def regime_of(z: float) -> int:
        if z <= bounds[0]:
            return 0
        if z <= bounds[1]:
            return 1
        return 2

    # bucket training events
    buckets: dict[tuple[str, int], list[int]] = {}
    for eid in train:
        key = (events[eid]["setup"], regime_of(float(events[eid]["gk_vol_z"])))
        buckets.setdefault(key, []).append(eid)

    # pooled fallback learned on everything
    fallback = _search_cell(cfg, grid, scans, train, "*", -1)

    cells: dict[tuple[str, int], GeoCell] = {}
    for (setup, regime), ids in buckets.items():
        if len(ids) >= min_samples:
            cells[(setup, regime)] = _search_cell(cfg, grid, scans, ids, setup, regime)
        else:
            # not enough evidence in the cell — inherit the pooled geometry
            cells[(setup, regime)] = GeoCell(
                setup, regime, fallback.stop_atr, fallback.target_atr,
                fallback.p_win, fallback.expectancy_r, len(ids))
    for c in cells.values():
        log.info("geometry %s r%d -> stop %.2f ATR / target %.2f ATR "
                 "(p=%.3f, E=%.3fR, n=%d)",
                 c.setup, c.regime, c.stop_atr, c.target_atr,
                 c.p_win, c.expectancy_r, c.samples)
    return Geometry(cells, bounds, fallback)


def _search_cell(cfg: Config, grid: np.ndarray, scans: dict[int, PathScan],
                 ids: list[int], setup: str, regime: int) -> GeoCell:
    gcfg = cfg.geometry
    adv = np.stack([scans[i].adv_cross for i in ids])   # (n, G)
    fav = np.stack([scans[i].fav_cross for i in ids])
    eod = np.array([scans[i].eod_signed_atr for i in ids])
    n = len(ids)

    # candidate levels straight from the observed excursion distributions
    max_adv = np.array([_max_level(adv[k], grid) for k in range(n)])
    max_fav = np.array([_max_level(fav[k], grid) for k in range(n)])
    stop_cands = _snap(np.quantile(max_adv, gcfg["stop_quantiles"]), grid)
    tgt_cands = _snap(np.quantile(max_fav, gcfg["target_quantiles"]), grid)

    best: GeoCell | None = None
    for s in stop_cands:
        si = int(np.argmin(np.abs(grid - s)))
        a = adv[:, si]
        for t in tgt_cands:
            if s <= 0 or t <= 0:
                continue
            ti = int(np.argmin(np.abs(grid - t)))
            f = fav[:, ti]
            win = (f != -1) & ((a == -1) | (f < a))     # ties -> loss
            stopped = (a != -1) & ~win
            neither = ~win & ~stopped
            p_win = float(win.mean())
            p_stop = float(stopped.mean())
            eod_r = float(np.mean(eod[neither]) / s) if neither.any() else 0.0
            exp = p_win * (t / s) - p_stop + float(neither.mean()) * eod_r
            if best is None or exp > best.expectancy_r:
                best = GeoCell(setup, regime, float(s), float(t),
                               p_win, float(exp), n)
    assert best is not None
    return best


def _max_level(cross_row: np.ndarray, grid: np.ndarray) -> float:
    hit = cross_row != -1
    return float(grid[hit].max()) if hit.any() else float(grid[0])


def _snap(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    snapped = np.array([grid[int(np.argmin(np.abs(grid - v)))] for v in vals])
    return np.unique(snapped)
