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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

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
    auc_test: float | None = None   # ROC AUC on the test fold (None if single-class)
    auc_val: float | None = None    # ROC AUC on the train-tail validation slice
    prob_hist: list[int] | None = None  # 20 fixed bins over [0, 1] of test probs


def _safe_auc(y: np.ndarray, probs: np.ndarray) -> float | None:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probs))


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

        # chronological tail 20% of train = internal validation slice, shared
        # by the calibrator and the threshold picker (both see ONLY this slice)
        cut = int(train.height * 0.8)
        val = train.sort("entry_ts").tail(train.height - cut)
        val_raw = (clf.predict_proba(val.select(FEATURES).to_numpy())[:, 1]
                   if val.height else np.array([]))
        y_val = val["outcome"].to_numpy() if val.height else np.array([])

        calibrator = None
        if (bool(mcfg.get("calibrate", False)) and val.height >= 10
                and len(np.unique(y_val)) > 1):
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_raw, y_val)
        val_probs = (np.asarray(calibrator.predict(val_raw), dtype=float)
                     if calibrator is not None else val_raw)

        threshold = _pick_threshold(cfg, val_probs, val)

        raw = clf.predict_proba(test.select(FEATURES).to_numpy())[:, 1]
        probs = (np.asarray(calibrator.predict(raw), dtype=float)
                 if calibrator is not None else raw)
        test = test.with_columns(
            pl.Series("prob", probs.astype(float)),
            pl.lit(float(threshold)).alias("threshold"),
            pl.lit(m).alias("fold"),
        )
        auc_test = _safe_auc(test["outcome"].to_numpy(), probs)
        # rank on RAW tail probs: isotonic was fit on this slice, so its own
        # fitted values would report a self-fit-inflated AUC
        auc_val = _safe_auc(y_val, val_raw)
        prob_hist = np.histogram(probs, bins=20, range=(0.0, 1.0))[0].astype(int).tolist()

        gain = clf.booster_.feature_importance(importance_type="gain")
        imp = {f: float(g) for f, g in zip(FEATURES, gain)}
        results.append(FoldResult(m, test, float(threshold), train.height, imp,
                                  auc_test=auc_test, auc_val=auc_val,
                                  prob_hist=prob_hist))
        log.info("fold %s: train=%d test=%d thr=%.3f auc_test=%s auc_val=%s",
                 m, train.height, test.height, threshold,
                 f"{auc_test:.3f}" if auc_test is not None else "-",
                 f"{auc_val:.3f}" if auc_val is not None else "-")
    return results


def _pick_threshold(cfg: Config, probs: np.ndarray, val: pl.DataFrame) -> float:
    """Pick the threshold maximizing realized expectancy on the train tail.

    `probs` are the (calibrated, when enabled) probabilities for `val` — the
    same values the decision layer will compare against this threshold.
    """
    tcfg = cfg.model["threshold"]
    default = float(tcfg["default"])
    g = tcfg["grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 3)
    min_trades = int(tcfg["min_trades"])

    if val.height < min_trades:
        return default
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
