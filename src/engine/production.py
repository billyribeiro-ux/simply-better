"""Production trainer: one decision stack for live signal generation.

The walk-forward loop is a VALIDATION device — it exists to measure the edge
without lookahead. At live time nothing is future, so the production model
legitimately trains on all cached history through ``d_to``:

- attribution rules come from the same fold loop the research run uses
  (``learn.run_research``), so every rule was mined strictly out-of-sample;
- geometry is learned on every confirmed event (``geometry.learn`` with all
  ids) — the same code path each fold uses, widened to full history;
- the classifier/calibrator/threshold come from ``model.fit_scored_model`` —
  byte-for-byte the function every walk-forward fold uses.

Train/serve parity is asserted by tests/test_live_path.py: a production
bundle trained to a fold boundary reproduces that fold's probabilities and
decisions exactly.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from . import geometry, labeling, learn, model
from .attribution import AdjustmentRule
from .config import Config
from .features import FEATURES

log = logging.getLogger("engine.production")


@dataclass
class ProductionBundle:
    fm: model.FittedModel
    geometry: geometry.Geometry
    rules: list[AdjustmentRule]
    trained_from: date
    trained_through: date
    universe: list[str]
    features: list[str]
    created_at: str
    n_events: int
    n_labeled: int


def train_production(cfg: Config, d_from: date, d_to: date) -> ProductionBundle:
    state = learn.run_research(cfg, d_from, d_to)

    geo = geometry.learn(cfg, state.events, state.scans, state.all_ids)
    labeled = labeling.label(cfg, state.events, state.scans, geo.lookup)
    if labeled.height < 100:
        raise RuntimeError(
            f"only {labeled.height} labeled trades through {d_to} — "
            "not enough history to train a production model")
    labeled = labeled.sort("entry_ts")
    if labeled["outcome"].n_unique() < 2:
        raise RuntimeError("labeled history is single-class — cannot train")

    fm = model.fit_scored_model(cfg, labeled)
    log.info("production model: %d events, %d labeled, thr=%.3f, %d rules",
             len(state.events), labeled.height, fm.threshold,
             len(state.attribution.rules))
    return ProductionBundle(
        fm=fm, geometry=geo, rules=list(state.attribution.rules),
        trained_from=d_from, trained_through=d_to,
        universe=cfg.universe, features=list(FEATURES),
        created_at=datetime.now(timezone.utc).isoformat(),
        n_events=len(state.events), n_labeled=labeled.height,
    )


def bundle_path(cfg: Config) -> Path:
    return cfg.root / cfg.raw.get("live", {}).get(
        "model_path", "data/model/production.pkl")


def save_bundle(cfg: Config, bundle: ProductionBundle) -> Path:
    p = bundle_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(bundle, fh)
    meta = {
        "created_at": bundle.created_at,
        "trained_from": bundle.trained_from.isoformat(),
        "trained_through": bundle.trained_through.isoformat(),
        "universe": bundle.universe,
        "features": bundle.features,
        "threshold": bundle.fm.threshold,
        "n_events": bundle.n_events,
        "n_labeled": bundle.n_labeled,
        "rules": len(bundle.rules),
        "geometry": bundle.geometry.rows(),
    }
    p.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log.info("saved production bundle -> %s", p)
    return p


def load_bundle(cfg: Config) -> ProductionBundle:
    p = bundle_path(cfg)
    if not p.exists():
        raise RuntimeError(
            f"no production model at {p} — run: mie train --from ... --to ...")
    with p.open("rb") as fh:
        bundle: ProductionBundle = pickle.load(fh)
    if list(bundle.features) != list(FEATURES):
        raise RuntimeError(
            "production bundle feature set does not match the running engine "
            "(the code changed since training) — retrain with: mie train")
    return bundle
