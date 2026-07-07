"""Orchestration: the complete self-learning research loop.

Chronological, fold-by-fold:

  detect setups (5m) -> confirm entries + scan paths (1m)
  for each walk-forward month:
      learn geometry on train events only
      label train + fold with that geometry
      fit meta-model on train, pick threshold on train tail
      apply accumulated attribution rules (learned on PRIOR folds only)
      decide which fold trades are taken
      mine new loss-attribution rules from taken history
  simulate portfolio over all taken trades -> metrics -> artifacts

Nothing in any fold's decision path has seen that fold's outcomes.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from . import attribution as attr
from . import backtest, geometry, labeling, metrics, model, setups
from .anchors import ROOM_FEATURE
from .config import Config
from .data import Store
from .features import daily_context

log = logging.getLogger("engine.learn")


@dataclass
class ResearchState:
    """Everything the fold loop produced — consumed by run_pipeline's
    artifact builder and by production.train_production (which needs the
    honestly-accumulated attribution rules and the event/scan universe)."""
    events: list[dict]
    scans: dict
    all_ids: list[int]
    attribution: attr.Attribution
    fold_results: list = field(default_factory=list)
    all_decided: pl.DataFrame | None = None
    taken: pl.DataFrame | None = None
    geo_final: geometry.Geometry | None = None
    n_trials: int = 0


def run_pipeline(cfg: Config, d_from: date, d_to: date) -> dict:
    """Full research loop + artifact build (behavior unchanged)."""
    state = run_research(cfg, d_from, d_to)
    return _build_artifacts(cfg, state, d_from, d_to)


def run_research(cfg: Config, d_from: date, d_to: date) -> ResearchState:
    store = Store(cfg)
    detect_tf = cfg.raw["data"]["detect_tf"]
    refine_tf = cfg.raw["data"]["refine_tf"]

    # ---- 1. detect --------------------------------------------------------
    events: list[dict] = []
    bars1: dict[str, pl.DataFrame] = {}
    for sym in cfg.universe:
        ctx = daily_context(store.load_daily(sym))
        b5 = store.load_bars(detect_tf, sym, d_from, d_to)
        events.extend(setups.detect(cfg, sym, b5, ctx))
        bars1[sym] = store.load_bars(refine_tf, sym, d_from, d_to)
    if not events:
        raise RuntimeError("no setup events detected — check data cache & range")
    events.sort(key=lambda e: e["trigger_ts"])
    log.info("total events: %d", len(events))

    # ---- 2. confirm + path scan (1m) --------------------------------------
    scans = labeling.scan_paths(cfg, events, bars1)
    if not scans:
        raise RuntimeError("no entries confirmed on 1-min — check 1-min cache depth")

    # ---- 3. provisional labels (pooled geometry) for fold building --------
    all_ids = sorted(scans.keys())
    geo0 = geometry.learn(cfg, events, scans, all_ids)  # only to enumerate months
    provisional = labeling.label(cfg, events, scans, geo0.lookup)

    months = (provisional.with_columns(
        pl.col("session_date").dt.strftime("%Y-%m").alias("m"))["m"]
        .unique().sort().to_list())
    min_train = int(cfg.model["walk_forward"]["min_train_months"])
    if len(months) <= min_train:
        raise RuntimeError(
            f"{len(months)} months available, need > {min_train} — widen range")

    # ---- 4. fold loop with strictly-OOS geometry, model, and rules --------
    attribution = attr.Attribution()
    fold_results: list[model.FoldResult] = []
    taken_frames: list[pl.DataFrame] = []
    geo_final: geometry.Geometry = geo0
    n_trials = 0

    ids_by_month: dict[str, list[int]] = {}
    for eid in all_ids:
        m = f"{events[eid]['session_date']:%Y-%m}"
        ids_by_month.setdefault(m, []).append(eid)

    for k in range(min_train, len(months)):
        fold_m = months[k]
        train_ids = [i for mm in months[:k] for i in ids_by_month.get(mm, [])]
        fold_ids = ids_by_month.get(fold_m, [])
        if len(train_ids) < 100 or not fold_ids:
            continue

        geo = geometry.learn(cfg, events, scans, train_ids)
        geo_final = geo
        # deflation pays for the FULL search width: per cell, every stop
        # candidate crossed with every atr target candidate plus every
        # (anchor x frac) variant; plus the threshold grid per fold
        gcfg = cfg.geometry
        n_geo_variants = len(gcfg["stop_quantiles"]) * (
            len(gcfg["target_quantiles"])
            + len(gcfg["anchors"]) * len(gcfg["anchor_fracs"])
        )
        thr_g = cfg.model["threshold"]["grid"]
        thr_n = len(np.arange(thr_g["start"], thr_g["stop"] + 1e-9, thr_g["step"]))
        n_trials += len(geo.cells) * n_geo_variants + thr_n

        labeled = labeling.label(
            cfg,
            [events[i] for i in train_ids + fold_ids],
            {j: scans[eid] for j, eid in enumerate(train_ids + fold_ids)},
            geo.lookup,
        )
        # walk_forward internally splits by month; restrict to this fold's step
        frs = [fr for fr in model.walk_forward(cfg, labeled, only_fold=fold_m)
               if fr.fold == fold_m]
        if not frs:
            continue
        fr = frs[0]
        fold_results.append(fr)

        # decide takes with attribution-adjusted thresholds (rules < fold only)
        # plus the day-type trend veto (frozen config, applied identically in
        # live.scan_live — an execution-time gate, never a detection filter)
        gate_max = float(cfg.setups.get("day_type", {}).get("eff_gate_max", 1e9))
        take_rows = []
        for row in fr.test.iter_rows(named=True):
            eff = attribution.effective_threshold(row, fr.threshold, fold_m)
            row["threshold"] = eff
            trend_veto = float(row.get("dt_eff_h1_aligned", 0.0)) > gate_max
            row["taken"] = bool(row["prob"] >= eff and not trend_veto)
            take_rows.append(row)
        fold_df = pl.DataFrame(take_rows)
        taken_frames.append(fold_df)

        # learn new rules from everything decided so far (this fold included —
        # rules carry learned_after_fold=fold_m so they only fire on later folds)
        hist = pl.concat(taken_frames, how="vertical_relaxed").filter(pl.col("taken"))
        if hist.height > 0:
            attr.learn_rules(cfg, hist, fold_m, attribution)

    if not taken_frames:
        raise RuntimeError("walk-forward produced no folds — widen the range")

    # research-program searches outside this loop also mined this range —
    # charge them into the deflation once (config-declared ledger)
    n_trials += int(cfg.model.get("external_trials", 0))

    all_decided = pl.concat(taken_frames, how="vertical_relaxed")
    taken = all_decided.filter(pl.col("taken"))
    log.info("decided=%d taken=%d rules=%d",
             all_decided.height, taken.height, len(attribution.rules))

    return ResearchState(
        events=events, scans=scans, all_ids=all_ids, attribution=attribution,
        fold_results=fold_results, all_decided=all_decided, taken=taken,
        geo_final=geo_final, n_trials=n_trials,
    )


def _build_artifacts(cfg: Config, state: ResearchState,
                     d_from: date, d_to: date) -> dict:
    store = Store(cfg)
    all_decided, taken = state.all_decided, state.taken
    fold_results, attribution = state.fold_results, state.attribution
    geo_final, n_trials = state.geo_final, state.n_trials

    # ---- 5. portfolio simulation ------------------------------------------
    trades, eq_dates, eq_curve = backtest.run(cfg, taken)
    summary = metrics.summarize(trades, eq_dates, eq_curve, max(n_trials, 1))

    clusters = attr.cluster_losses(trades) if trades.height else []
    per_symbol = _breakdown(trades, "symbol")
    per_setup = _breakdown(trades, "setup")
    diagnostics = _diagnostics(fold_results)
    anchor_recovery = _anchor_recovery(all_decided)

    artifacts = {
        "run_id": uuid.uuid4().hex[:12],
        "run_type": "exploratory",
        "from": d_from.isoformat(),
        "to": d_to.isoformat(),
        "universe": cfg.universe,
        "summary": summary,
        "per_symbol": per_symbol,
        "per_setup": per_setup,
        "feature_importance": model.aggregate_importance(fold_results),
        "folds": len(fold_results),
        "diagnostics": diagnostics,
        "anchor_recovery": anchor_recovery,
        "geometry": geo_final.rows(),
        "attribution": {
            "rules": attribution.rules_json(),
            "clusters": clusters,
        },
        "equity": {
            "dates": [d.isoformat() for d in eq_dates],
            "curve": [round(float(x), 2) for x in eq_curve],
        },
        "signals": _signals_json(all_decided, trades),
    }

    store.persist_run(
        artifacts["run_id"], trades,
        json.dumps({k: artifacts[k] for k in
                    ("from", "to", "summary", "per_symbol", "per_setup",
                     "feature_importance", "folds")}),
        json.dumps(artifacts["geometry"]),
        json.dumps(artifacts["attribution"]),
    )
    return artifacts


def _diagnostics(fold_results: list[model.FoldResult]) -> dict:
    """Per-fold AUC + probability-distribution record for the dashboard."""
    folds = [
        {
            "fold": fr.fold,
            "auc_test": round(fr.auc_test, 4) if fr.auc_test is not None else None,
            "auc_val": round(fr.auc_val, 4) if fr.auc_val is not None else None,
            "threshold": round(fr.threshold, 3),
            "n_test": fr.test.height,
            "prob_hist": fr.prob_hist or [],
        }
        for fr in fold_results
    ]
    aucs = [fr.auc_test for fr in fold_results if fr.auc_test is not None]
    return {
        "folds": folds,
        "auc_mean_test": round(float(np.mean(aucs)), 4) if aucs else None,
    }


def _anchor_recovery(decided: pl.DataFrame) -> list[dict]:
    """Fraction of anchor distance recovered, per (setup, anchor).

    recovery_i = mfe_r_i / max(room_i / stop_atr_i, eps) — how much of the
    runway to each anchor the trade actually traversed. Median and p75 over
    decided trades with positive room. This is the empirical answer to
    "reversion to what?".
    """
    if decided.height == 0:
        return []
    eps = 1e-9
    out: list[dict] = []
    for setup in decided["setup"].unique().sort().to_list():
        sd = decided.filter(pl.col("setup") == setup)
        for anchor, key in ROOM_FEATURE.items():
            if key not in sd.columns:
                continue
            g = sd.filter(pl.col(key) > 0)
            if g.height == 0:
                continue
            room_r = (g[key] / g["stop_atr"]).clip(eps)
            rec = (g["mfe_r"] / room_r)
            out.append({
                "setup": setup, "anchor": anchor,
                "median": round(float(rec.median()), 3),
                "p75": round(float(rec.quantile(0.75)), 3),
                "samples": int(g.height),
            })
    return out


def _breakdown(trades: pl.DataFrame, col: str) -> list[dict]:
    if trades.height == 0:
        return []
    out = []
    for (val,), g in trades.group_by([col], maintain_order=False):
        pnl = g["pnl_usd"].to_numpy()
        out.append({
            col: val, "trades": int(g.height),
            "win_rate": round(float((pnl > 0).mean()) * 100, 2),
            "net_pnl_usd": round(float(pnl.sum()), 2),
        })
    return sorted(out, key=lambda r: -r["net_pnl_usd"])


def _signals_json(decided: pl.DataFrame, trades: pl.DataFrame) -> list[dict]:
    pnl_map: dict = {}
    shares_map: dict = {}
    if trades.height:
        for r in trades.iter_rows(named=True):
            pnl_map[r["event_id"]] = r["pnl_usd"]
            shares_map[r["event_id"]] = r["shares"]
    out = []
    for r in decided.sort("entry_ts").iter_rows(named=True):
        out.append({
            "id": int(r["event_id"]),
            "symbol": r["symbol"],
            "date": r["session_date"].isoformat(),
            "setup": r["setup"],
            "side": "SHORT" if r["side"] < 0 else "LONG",
            "trigger_ts": r["trigger_ts"].isoformat(),
            "entry_ts": r["entry_ts"].isoformat(),
            "entry_px": round(float(r["entry_px"]), 2),
            "stop_px": round(float(r["stop_px"]), 2),
            "target_px": round(float(r["target_px"]), 2),
            "exit_ts": r["exit_ts"].isoformat(),
            "exit_px": round(float(r["exit_px"]), 2),
            "exit_reason": r["exit_reason"],
            "prob": round(float(r["prob"]), 3),
            "threshold": round(float(r["threshold"]), 3),
            "taken": bool(r["taken"]),
            "outcome": "WIN" if r["outcome"] == 1 else "LOSS",
            "pnl_r": round(float(r["pnl_r"]), 2),
            "pnl_usd": pnl_map.get(r["event_id"]),
            "shares": shares_map.get(r["event_id"]),
            "mae_r": round(float(r["mae_r"]), 2),
            "mfe_r": round(float(r["mfe_r"]), 2),
        })
    return out
