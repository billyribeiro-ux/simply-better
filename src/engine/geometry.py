"""Learned trade geometry.

For each (setup, vol_regime) cell, search stop/target candidates drawn from
the empirical MAE/MFE distributions of the TRAINING events only, and pick the
variant that maximizes THROUGHPUT-ADJUSTED expectancy in R:

    score = E[R | traded] * (n_traded / n_events)

Two families of target compete per cell:

- "atr" mode (iteration one): fixed ATR-multiple targets from MFE quantiles.
  Every event trades, so throughput = 1 and score = per-trade expectancy
      E = p_win * (t/s) + p_stop * (-1) + p_eod * E[eod_r | eod]
- anchor modes: per-event target t_i = clamp(frac * room_anchor_i, grid
  bounds) snapped to the scan grid. Events with room below
  ``geometry.min_room_atr`` are NO-TRADE under that variant — they reduce
  throughput, they are not counted as losses. A variant that only trades 30%
  of events must therefore beat "atr" on score, not per-trade expectancy.

Ties break toward "atr" (the simpler model) because atr candidates are
evaluated first under a strictly-greater comparison.

Because ``PathScan`` already holds first-crossing indices for the whole grid,
every candidate variant is evaluated with pure array comparisons — the exact
same path data, zero re-simulation, zero lookahead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .anchors import ROOM_FEATURE
from .config import Config
from .labeling import PathScan

log = logging.getLogger("engine.geometry")


@dataclass(frozen=True)
class GeoCell:
    setup: str
    regime: int
    stop_atr: float
    target_atr: float      # fixed level in atr mode; representative only otherwise
    p_win: float           # over traded events
    expectancy_r: float    # throughput-adjusted score (see module docstring)
    samples: int
    anchor: str = "atr"    # "atr" = iteration-one behavior
    frac: float = 0.0      # fraction of anchor distance (0.0 in atr mode)


class Geometry:
    def __init__(self, cells: dict[tuple[str, int], GeoCell],
                 regime_bounds: tuple[float, float], fallback: GeoCell,
                 grid: np.ndarray | None = None,
                 min_room_atr: float = 0.3) -> None:
        self.cells = cells
        self.regime_bounds = regime_bounds
        self.fallback = fallback
        self.grid = grid
        self.min_room_atr = float(min_room_atr)

    def regime_of(self, gk_vol_z: float) -> int:
        lo, hi = self.regime_bounds
        if gk_vol_z <= lo:
            return 0
        if gk_vol_z <= hi:
            return 1
        return 2

    def lookup(self, ev: dict) -> tuple[float, float, bool]:
        """(stop_atr, target_atr, tradable) for one event.

        atr cells trade everything; anchor cells derive the target from the
        event's own room and refuse events without ``min_room_atr`` runway.
        """
        cell = self.cells.get((ev["setup"], self.regime_of(float(ev["gk_vol_z"]))),
                              self.fallback)
        if cell.anchor == "atr" or self.grid is None:
            return cell.stop_atr, cell.target_atr, True
        room = float(ev.get(ROOM_FEATURE[cell.anchor], 0.0))
        tradable = room >= self.min_room_atr
        t = _snap_level(cell.frac * room, self.grid)
        return cell.stop_atr, t, tradable

    def rows(self) -> list[dict]:
        return [
            {"setup": c.setup, "regime": c.regime, "anchor": c.anchor,
             "frac": c.frac, "stop_atr": c.stop_atr,
             "target_atr": c.target_atr, "p_win": round(c.p_win, 4),
             "expectancy_r": round(c.expectancy_r, 4), "samples": c.samples}
            for c in sorted(self.cells.values(), key=lambda x: (x.setup, x.regime))
        ]


def _snap_level(t: float, grid: np.ndarray) -> float:
    """Clamp to grid bounds and snap to the (uniform) scan grid."""
    step = float(grid[1] - grid[0])
    idx = int(round((min(max(t, float(grid[0])), float(grid[-1])) - float(grid[0])) / step))
    return float(grid[max(0, min(idx, len(grid) - 1))])


def learn(cfg: Config, events: list[dict], scans: dict[int, PathScan],
          train_ids: list[int]) -> Geometry:
    grid = cfg.scan_grid
    gcfg = cfg.geometry
    min_samples = int(gcfg["min_samples"])

    min_room = float(gcfg.get("min_room_atr", 0.3))

    train = [eid for eid in train_ids if eid in scans]
    if not train:
        fb = GeoCell("*", -1, 0.5, 0.75, 0.0, 0.0, 0)
        return Geometry({}, (-0.43, 0.43), fb, grid=grid, min_room_atr=min_room)

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
    fallback = _search_cell(cfg, grid, scans, train, "*", -1, events=events)

    cells: dict[tuple[str, int], GeoCell] = {}
    for (setup, regime), ids in buckets.items():
        if len(ids) >= min_samples:
            cells[(setup, regime)] = _search_cell(cfg, grid, scans, ids,
                                                  setup, regime, events=events)
        else:
            # not enough evidence in the cell — inherit the pooled geometry
            cells[(setup, regime)] = GeoCell(
                setup, regime, fallback.stop_atr, fallback.target_atr,
                fallback.p_win, fallback.expectancy_r, len(ids),
                anchor=fallback.anchor, frac=fallback.frac)
    for c in cells.values():
        log.info("geometry %s r%d -> %s stop %.2f ATR / target %.2f "
                 "(frac=%.3f, p=%.3f, score=%.3fR, n=%d)",
                 c.setup, c.regime, c.anchor, c.stop_atr, c.target_atr,
                 c.frac, c.p_win, c.expectancy_r, c.samples)
    return Geometry(cells, bounds, fallback, grid=grid, min_room_atr=min_room)


def _search_cell(cfg: Config, grid: np.ndarray, scans: dict[int, PathScan],
                 ids: list[int], setup: str, regime: int,
                 events: list[dict] | None = None) -> GeoCell:
    """Search atr-mode and (when events are provided) anchor-mode variants.

    Selection metric is throughput-adjusted expectancy:
        score = E[R | traded] * (n_traded / n_events)
    atr mode trades every event (throughput 1), anchor variants exclude
    events with room < min_room_atr as NO-TRADE — reduced throughput, not
    losses. ``events=None`` reproduces iteration-one atr-only behavior.
    """
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

    # ---- anchor modes: per-event targets at frac x room ------------------
    if events is not None:
        min_room = float(gcfg.get("min_room_atr", 0.3))
        step = float(grid[1] - grid[0])
        g0, g_max = float(grid[0]), float(grid[-1])
        rows_idx = np.arange(n)
        for anchor in gcfg.get("anchors", []):
            key = ROOM_FEATURE[anchor]
            rooms = np.array([float(events[i].get(key, 0.0)) for i in ids])
            traded = rooms >= min_room
            nt = int(traded.sum())
            if nt == 0:
                continue
            for f in gcfg.get("anchor_fracs", []):
                t_lvls = np.clip(float(f) * rooms, g0, g_max)
                ti = np.clip(np.round((t_lvls - g0) / step).astype(int),
                             0, len(grid) - 1)
                t_snap = grid[ti]
                fvals = fav[rows_idx, ti]      # each event's own target column
                for s in stop_cands:
                    if s <= 0:
                        continue
                    si = int(np.argmin(np.abs(grid - s)))
                    a = adv[:, si]
                    win = traded & (fvals != -1) & ((a == -1) | (fvals < a))
                    stopped = traded & (a != -1) & ~win
                    neither = traded & ~win & ~stopped
                    payoff = float((t_snap[win] / s).sum())
                    eod_sum = float((eod[neither] / s).sum())
                    exp_trade = (payoff - float(stopped.sum()) + eod_sum) / nt
                    score = exp_trade * (nt / n)
                    if score > best.expectancy_r:   # ties keep the simpler atr
                        best = GeoCell(
                            setup, regime, float(s),
                            float(np.median(t_snap[traded])),  # representative
                            float(win.sum() / nt), float(score), n,
                            anchor=anchor, frac=float(f))
    return best


def _max_level(cross_row: np.ndarray, grid: np.ndarray) -> float:
    hit = cross_row != -1
    return float(grid[hit].max()) if hit.any() else float(grid[0])


def _snap(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    snapped = np.array([grid[int(np.argmin(np.abs(grid - v)))] for v in vals])
    return np.unique(snapped)
