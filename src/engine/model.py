"""Meta-model: probability that a triggered setup reaches its target first.

Validation discipline:
- monthly walk-forward folds, chronological, expanding window
- embargo gap between train end and test start (Lopez de Prado purging;
  trades are intraday so the embargo alone removes all overlap)
- decision threshold picked on the chronological TAIL of the training set by
  maximizing realized expectancy, never on the fold being predicted
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier

from .config import Config
from .features import FEATURES

log = logging.getLogger("engine.model")

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names")


@dataclass
class FoldResult:
    fold: str                 # "YYYY-MM"
    test: pl.DataFrame        # labeled trades + prob + threshold
    threshold: float
    n_train: int
    importance: dict[str, float]


def walk_forward(cfg: Config, labeled: pl.DataFrame) -> list[FoldResult]:
    if labeled.height == 0:
        return []
    mcfg = cfg.model
    wf = mcfg["walk_forward"]
    embargo = timedelta(days=int(wf["embargo_days"]))
    min_train_months = int(wf["min_train_months"])

    df = labeled.sort("entry_ts").with_columns(
        pl.col("session_date").dt.strftime("%Y-%m").alias("fold_month")
    )
    months = df["fold_month"].unique().sort().to_list()
    if len(months) <= min_train_months:
        raise RuntimeError(
            f"only {len(months)} months of events — need > {min_train_months} "
            "for walk-forward. Widen the date range."
        )

    results: list[FoldResult] = []
    for m in months[min_train_months:]:
        month_start = df.filter(pl.col("fold_month") == m)["session_date"].min()
        train = df.filter(pl.col("session_date") < (month_start - embargo))
        test = df.filter(pl.col("fold_month") == m)
        if train.height < 100 or test.height == 0:
            continue

        X_tr = train.select(FEATURES).to_numpy()
        y_tr = train["outcome"].to_numpy()
        if len(np.unique(y_tr)) < 2:
            continue

        clf = LGBMClassifier(**mcfg["params"], verbose=-1)
        clf.fit(X_tr, y_tr)

        threshold = _pick_threshold(cfg, clf, train)

        probs = clf.predict_proba(test.select(FEATURES).to_numpy())[:, 1]
        test = test.with_columns(
            pl.Series("prob", probs.astype(float)),
            pl.lit(float(threshold)).alias("threshold"),
            pl.lit(m).alias("fold"),
        )
        gain = clf.booster_.feature_importance(importance_type="gain")
        imp = {f: float(g) for f, g in zip(FEATURES, gain)}
        results.append(FoldResult(m, test, float(threshold), train.height, imp))
        log.info("fold %s: train=%d test=%d thr=%.3f",
                 m, train.height, test.height, threshold)
    return results


def _pick_threshold(cfg: Config, clf: LGBMClassifier, train: pl.DataFrame) -> float:
    tcfg = cfg.model["threshold"]
    default = float(tcfg["default"])
    g = tcfg["grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 3)
    min_trades = int(tcfg["min_trades"])

    # chronological tail 20% of train = internal validation slice
    n = train.height
    cut = int(n * 0.8)
    val = train.sort("entry_ts").tail(n - cut)
    if val.height < min_trades:
        return default
    probs = clf.predict_proba(val.select(FEATURES).to_numpy())[:, 1]
    pnl = val["pnl_r"].to_numpy()

    best_thr, best_exp = default, -np.inf
    for thr in grid:
        take = probs >= thr
        if take.sum() < min_trades:
            continue
        exp = float(pnl[take].mean())
        if exp > best_exp:
            best_exp, best_thr = exp, float(thr)
    return best_thr


def aggregate_importance(folds: list[FoldResult]) -> list[dict]:
    if not folds:
        return []
    total: dict[str, float] = {}
    for fr in folds:
        for k, v in fr.importance.items():
            total[k] = total.get(k, 0.0) + v
    s = sum(total.values()) or 1.0
    ranked = sorted(total.items(), key=lambda kv: -kv[1])
    return [{"feature": k, "gain": round(v / s, 4)} for k, v in ranked]
