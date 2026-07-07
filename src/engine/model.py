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
class FittedModel:
    """One trained decision stack: seed-bagged classifiers + calibrator +
    threshold.

    This is the SINGLE fitting path shared by the walk-forward research loop
    and the production live trainer — the live path cannot drift from what
    was validated because they are the same code.
    """
    clfs: list[LGBMClassifier]
    calibrator: IsotonicRegression | None
    threshold: float
    auc_val: float | None
    features: list[str]

    @property
    def clf(self) -> LGBMClassifier:
        return self.clfs[0]


def _fit_clfs(cfg: Config, X: np.ndarray, y: np.ndarray) -> list[LGBMClassifier]:
    """Seed-bagged LightGBM ensemble (variance reduction, same data)."""
    n_seeds = max(1, int(cfg.model.get("ensemble_seeds", 1)))
    clfs = []
    for s in range(n_seeds):
        clf = LGBMClassifier(**cfg.model["params"], verbose=-1, random_state=7 + s)
        clf.fit(X, y)
        clfs.append(clf)
    return clfs


def _predict(clfs: list[LGBMClassifier], X: np.ndarray) -> np.ndarray:
    return np.mean([c.predict_proba(X)[:, 1] for c in clfs], axis=0)


def _oof_probs(cfg: Config, train_sorted: pl.DataFrame
               ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities over the training window.

    Expanding chronological blocks: block k's probabilities come from an
    ensemble fitted ONLY on rows whose session_date precedes the block start
    by more than the embargo (same purge rule as the walk-forward). Block 0
    has no prior data and gets NaN. Returns (probs, valid_mask).

    These are the honest inputs for calibration and threshold selection —
    in-sample probabilities from a 400-tree booster are memorized (~AUC 1.0)
    and produce degenerate step calibrators and pinned thresholds.
    """
    mcfg = cfg.model
    K = max(2, int(mcfg.get("oof_folds", 4)))
    n = train_sorted.height
    probs = np.full(n, np.nan)
    ords = np.array([d.toordinal() for d in train_sorted["session_date"].to_list()])
    X_all = train_sorted.select(FEATURES).to_numpy()
    y_all = train_sorted["outcome"].to_numpy()
    embargo_days = int(mcfg["walk_forward"]["embargo_days"])
    edges = [int(round(i * n / K)) for i in range(K + 1)]

    for k in range(1, K):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        cutoff = ords[lo] - embargo_days
        n_fit = int(np.searchsorted(ords, cutoff, side="left"))
        if n_fit < 50 or len(np.unique(y_all[:n_fit])) < 2:
            continue
        clfs = _fit_clfs(cfg, X_all[:n_fit], y_all[:n_fit])
        probs[lo:hi] = _predict(clfs, X_all[lo:hi])
    return probs, ~np.isnan(probs)


def fit_scored_model(cfg: Config, train: pl.DataFrame) -> FittedModel:
    """Fit the ensemble on ALL of `train`; calibrate on the training window's
    out-of-fold probabilities; pick the threshold on the chronological tail
    20% (invariant 4) using those same honest OOF probabilities."""
    mcfg = cfg.model
    train = train.sort("entry_ts")
    X_tr = train.select(FEATURES).to_numpy()
    y_tr = train["outcome"].to_numpy()
    clfs = _fit_clfs(cfg, X_tr, y_tr)

    oof, valid = _oof_probs(cfg, train)
    n = train.height
    cut = int(n * 0.8)

    calibrator = None
    auc_val: float | None = None
    if valid.sum() >= 30 and len(np.unique(y_tr[valid])) > 1:
        auc_val = _safe_auc(y_tr[valid], oof[valid])
        if bool(mcfg.get("calibrate", False)):
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(oof[valid], y_tr[valid])

    # threshold: tail-20% rows that have OOF coverage, calibrated view
    tail_valid = valid.copy()
    tail_valid[:cut] = False
    if tail_valid.sum() >= 10:
        thr_probs = oof[tail_valid]
        if calibrator is not None:
            thr_probs = np.asarray(calibrator.predict(thr_probs), dtype=float)
        threshold = _pick_threshold(cfg, thr_probs, train.filter(pl.Series(tail_valid)))
    else:
        threshold = float(mcfg["threshold"]["default"])

    return FittedModel(clfs, calibrator, float(threshold), auc_val, list(FEATURES))


def score(fm: FittedModel, X: np.ndarray) -> np.ndarray:
    """Ensemble-mean probabilities passed through the fitted calibrator."""
    raw = _predict(fm.clfs, X)
    if fm.calibrator is not None:
        return np.asarray(fm.calibrator.predict(raw), dtype=float)
    return np.asarray(raw, dtype=float)


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


def walk_forward(cfg: Config, labeled: pl.DataFrame,
                 only_fold: str | None = None) -> list[FoldResult]:
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
        if only_fold is not None and m != only_fold:
            continue
        month_start = df.filter(pl.col("fold_month") == m)["session_date"].min()
        train = df.filter(pl.col("session_date") < (month_start - embargo))
        test = df.filter(pl.col("fold_month") == m)
        if train.height < 100 or test.height == 0:
            continue

        y_tr = train["outcome"].to_numpy()
        if len(np.unique(y_tr)) < 2:
            continue

        # shared fitting path: model on all of train, calibrator + threshold
        # on the chronological tail 20% (the only slice they ever see)
        fm = fit_scored_model(cfg, train)
        threshold = fm.threshold

        probs = score(fm, test.select(FEATURES).to_numpy())
        test = test.with_columns(
            pl.Series("prob", probs.astype(float)),
            pl.lit(float(threshold)).alias("threshold"),
            pl.lit(m).alias("fold"),
        )
        auc_test = _safe_auc(test["outcome"].to_numpy(), probs)
        auc_val = fm.auc_val
        prob_hist = np.histogram(probs, bins=20, range=(0.0, 1.0))[0].astype(int).tolist()

        gain = fm.clf.booster_.feature_importance(importance_type="gain")
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
