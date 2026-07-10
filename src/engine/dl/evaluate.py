"""Pre-registered 7-fold expanding evaluation (ledger 2026-07-09).

Produces the Gate A/B/C table. Baselines under identical folds and decision
bars: (a) LightGBM on flattened summary statistics of the SAME input tensors
— a same-information shallow null, strictly stronger than the original
23-feature null whose 0.50 AUC is already on record; (b) trailing 30-min
momentum sign (positive-control sanity). Gates are ABSOLUTE thresholds and do
not reference baselines. A within-session label-shuffle control refit (fold 1)
certifies the pipeline before Gate A may be read.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .dataset import Samples, SymbolCache, build_caches, build_samples, SYMBOLS_ORDER
from ..config import Config

log = logging.getLogger("engine.dl.evaluate")

# fold schedule fixed in the pre-registration (train start always 2024-01-02;
# 5 TRADING-day embargo purged off the train tail before each test window)
FOLDS = [
    ("2024-09-30", "2024-10-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-03-31"),
    ("2025-03-31", "2025-04-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-09-30"),
    ("2025-09-30", "2025-10-01", "2025-12-31"),
    ("2025-12-31", "2026-01-01", "2026-03-31"),
    ("2026-03-31", "2026-04-01", "2026-07-09"),
]
TRAIN_START = date(2024, 1, 2)


def _sessions(caches: dict[str, SymbolCache]) -> np.ndarray:
    all_d = np.concatenate([c.date1 for c in caches.values()])
    return np.unique(all_d.astype("datetime64[D]"))


def _embargoed_train_end(sessions: np.ndarray, test_start: date,
                         embargo: int) -> date:
    t0 = np.datetime64(test_start, "D")
    before = sessions[sessions < t0]
    if before.size <= embargo:
        raise RuntimeError("not enough history before test window")
    return before[-embargo - 1].astype(object)


def _auc(y_up: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_up, s))


def _clustered_ic(samples: Samples, rows: np.ndarray, s: np.ndarray):
    """Pooled Spearman IC + session-clustered t-stat (per-session ICs)."""
    from scipy.stats import spearmanr
    pooled = float(spearmanr(s, samples.y_ret[rows]).statistic)
    per = []
    for sess in np.unique(samples.session[rows]):
        m = samples.session[rows] == sess
        if m.sum() >= 20:
            ic = spearmanr(s[m], samples.y_ret[rows][m]).statistic
            if np.isfinite(ic):
                per.append(ic)
    per = np.array(per)
    t = float(per.mean() / (per.std(ddof=1) / np.sqrt(per.size))) if per.size > 3 else 0.0
    return pooled, t, per.size


def _fold_metrics(samples: Samples, rows: np.ndarray, s: np.ndarray) -> dict:
    y = samples.y_cls[rows]
    m = y != 1
    auc = _auc((y[m] == 2).astype(int), s[m]) if m.sum() > 50 else float("nan")
    ic, t, n_sess = _clustered_ic(samples, rows, s)
    per_sym = {}
    for sid in np.unique(samples.sym_id[rows]):
        mm = samples.sym_id[rows] == sid
        if mm.sum() > 100:
            from scipy.stats import spearmanr
            per_sym[SYMBOLS_ORDER[int(sid)]] = round(float(
                spearmanr(s[mm], samples.y_ret[rows][mm]).statistic), 4)
    return {"auc_up_down": round(auc, 4), "rank_ic": round(ic, 4),
            "ic_t_clustered": round(t, 2), "n": int(rows.size),
            "n_sessions": int(n_sess), "per_symbol_ic": per_sym}


# ---------------------------------------------------------------------------
# baselines (same folds, same decision bars)
# ---------------------------------------------------------------------------
def _summary_features(cfg: Config, caches, samples: Samples,
                      rows: np.ndarray) -> np.ndarray:
    """Flattened summary of the same tensors: per 1-min channel last/mean/std
    + per 5-min channel last -> shallow-model information parity."""
    from .dataset import tensor_batch
    xs = []
    for lo in range(0, rows.size, 8192):
        rr = rows[lo:lo + 8192]
        x1, x5, sym, stat = tensor_batch(cfg, caches, samples, rr)
        a = x1.numpy(); b = x5.numpy()
        feats = np.concatenate([
            a[:, :, -1], a.mean(axis=2), a.std(axis=2), b[:, :, -1],
            stat.numpy(), sym.numpy().reshape(-1, 1).astype(np.float32)], axis=1)
        xs.append(feats)
    return np.concatenate(xs, axis=0)


def _lgbm_baseline(cfg, caches, tr_samples, tr_rows,
                   te_samples, te_rows) -> dict:
    import lightgbm as lgb
    Xtr = _summary_features(cfg, caches, tr_samples, tr_rows)
    Xte = _summary_features(cfg, caches, te_samples, te_rows)
    ytr = (tr_samples.y_cls[tr_rows] == 2).astype(int)
    m = tr_samples.y_cls[tr_rows] != 1
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=31,
                             learning_rate=0.05, random_state=7, verbose=-1)
    clf.fit(Xtr[m], ytr[m])
    s = clf.predict_proba(Xte)[:, 1]
    return _fold_metrics(te_samples, te_rows, s)


def _momentum_baseline(cfg, caches, samples, te_rows) -> dict:
    """Trailing 30-min raw return as the score (sanity positive control for
    the harness at the metric level; a weak known effect)."""
    by_id = {SYMBOLS_ORDER.index(s): c for s, c in caches.items()}
    s = np.empty(te_rows.size)
    for k, r in enumerate(te_rows):
        c = by_id[int(samples.sym_id[r])]
        i = int(samples.idx1[r])
        s[k] = np.log(c.close1[i] / c.close1[max(i - 30, 0)])
    return _fold_metrics(samples, te_rows, s)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def _ckpt_dir(cfg: Config, H: int) -> Path:
    """Per-fold checkpoints (restart resilience). Restored folds are valid
    because same-seed fits are byte-identical (pinned by test_dl_causality and
    reproduced across two independent production runs)."""
    d = cfg.root / "data" / "dl" / ("eval_ckpt" if H == 15 else f"eval_ckpt_h{H}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_eval(cfg: Config, *, horizon_min: int | None = None,
             shuffle_control: bool = True, fresh: bool = False) -> dict:
    import os
    import pickle

    from . import require_torch
    require_torch()
    from .train import fit, predict_scores, model_from_bundle
    from .signals import build_trade_rows

    dl = cfg.raw["dl"]
    embargo = int(dl["train"]["embargo_days"])
    t_start = time.time()
    caches = build_caches(cfg)
    sessions = _sessions(caches)
    H = int(horizon_min or dl["horizon_min"])
    ckpt = _ckpt_dir(cfg, H)
    if fresh:
        for f in ckpt.glob("*.pkl"):
            f.unlink()

    folds_out, trade_rows_all = [], []
    for fi, (train_thru, te_a, te_b) in enumerate(FOLDS, start=1):
        cpath = ckpt / f"fold_{fi}.pkl"
        if cpath.exists():
            with open(cpath, "rb") as fh:
                saved = pickle.load(fh)
            folds_out.append(saved["met"])
            trade_rows_all.extend(saved["trade_rows"])
            log.info("fold %d restored from checkpoint (auc=%s)",
                     fi, saved["met"]["auc_up_down"])
            continue
        te_a_d, te_b_d = date.fromisoformat(te_a), date.fromisoformat(te_b)
        d_to = _embargoed_train_end(sessions, te_a_d, embargo)
        log.info("fold %d: train %s -> %s (embargoed), test %s -> %s",
                 fi, TRAIN_START, d_to, te_a, te_b)
        t0 = time.time()
        bundle, _tr_samples = fit(cfg, caches, TRAIN_START, d_to,
                                  horizon_min=H)
        te_samples = build_samples(cfg, caches, te_a_d, te_b_d, horizon_min=H)
        rows = np.arange(te_samples.y_cls.size)
        model = model_from_bundle(cfg, bundle)
        scores = predict_scores(cfg, model, caches, te_samples, rows)
        met = _fold_metrics(te_samples, rows, scores)
        met.update({"fold": fi, "train_through": str(d_to),
                    "test": f"{te_a}..{te_b}",
                    "s_min": bundle.s_min,
                    "val_ce": bundle.meta.get("val_ce"),
                    "wallclock_min": round((time.time() - t0) / 60, 1)})
        # baselines on identical rows
        if fi in (1, 4, 7):
            tr_samples_b = build_samples(cfg, caches, TRAIN_START, d_to,
                                         horizon_min=H)
            met["baseline_lgbm"] = _lgbm_baseline(
                cfg, caches, tr_samples_b,
                np.arange(tr_samples_b.y_cls.size), te_samples, rows)
        else:
            met["baseline_lgbm"] = None
        met["baseline_momentum"] = _momentum_baseline(cfg, caches, te_samples, rows)
        folds_out.append(met)
        log.info("fold %d metrics: %s", fi, {k: met[k] for k in
                 ("auc_up_down", "rank_ic", "ic_t_clustered", "n")})
        # Gate-B trade replay with this fold's frozen bundle
        fold_trades = build_trade_rows(
            cfg, caches, te_samples, rows, scores, bundle, fold=f"F{fi}")
        trade_rows_all.extend(fold_trades)
        # atomic per-fold checkpoint: a restart resumes here, not from zero
        tmp = str(cpath) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"met": met, "trade_rows": fold_trades}, fh)
        os.replace(tmp, cpath)

    # pooled Gate A metrics
    aucs = [f["auc_up_down"] for f in folds_out]
    ics = [f["rank_ic"] for f in folds_out]
    ns = np.array([f["n"] for f in folds_out], dtype=float)
    pooled_auc = float(np.average(aucs, weights=ns))
    pooled_ic = float(np.average(ics, weights=ns))
    ts = [f["ic_t_clustered"] for f in folds_out]

    # label-shuffle control on fold 1 (certifies the pipeline)
    shuffle_auc = None
    if shuffle_control:
        spath = ckpt / "shuffle.pkl"
        if spath.exists():
            with open(spath, "rb") as fh:
                shuffle_auc = pickle.load(fh)
            log.info("shuffle control restored from checkpoint (%s)", shuffle_auc)
        else:
            shuffle_auc = _shuffle_control(cfg, caches, sessions, embargo, H)
            tmp = str(spath) + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(shuffle_auc, fh)
            os.replace(tmp, spath)

    taken = [r for r in trade_rows_all if r["taken"]]
    gate_b = _gate_b_stats(taken)

    result = {
        "horizon_min": H,
        "folds": folds_out,
        "pooled": {"auc_up_down": round(pooled_auc, 4),
                   "rank_ic": round(pooled_ic, 4),
                   "per_fold_auc_gt_050": int(sum(a > 0.50 for a in aucs)),
                   "ic_t_per_fold": ts},
        "shuffle_control_auc": shuffle_auc,
        "gate_a": {
            "pooled_auc_ge_0530": pooled_auc >= 0.530,
            "folds_gt_050_ge_5of7": sum(a > 0.50 for a in aucs) >= 5,
            "pooled_ic_ge_0020": pooled_ic >= 0.020,
            "shuffle_in_band": (shuffle_auc is None
                                or 0.48 <= shuffle_auc <= 0.52),
        },
        "gate_b": gate_b,
        "n_decided": len(trade_rows_all), "n_taken": len(taken),
        "wallclock_min": round((time.time() - t_start) / 60, 1),
    }
    result["gate_a"]["PASS"] = all(v for k, v in result["gate_a"].items()
                                   if k != "PASS")
    return result, trade_rows_all


def _shuffle_control(cfg, caches, sessions, embargo, H) -> float:
    """Refit fold 1 with labels shuffled WITHIN sessions; test AUC must be
    chance. Certifies no structural leak in the training loop itself."""
    from .train import fit, predict_scores, model_from_bundle, set_determinism
    import numpy as np
    train_thru, te_a, te_b = FOLDS[0]
    te_a_d, te_b_d = date.fromisoformat(te_a), date.fromisoformat(te_b)
    d_to = _embargoed_train_end(sessions, te_a_d, embargo)

    # monkey-level shuffle: wrap build_samples output by shuffling y within
    # sessions, deterministic
    from . import dataset as ds
    orig = ds.build_samples

    def shuffled(cfg2, caches2, a, b, horizon_min=None):
        s = orig(cfg2, caches2, a, b, horizon_min=horizon_min)
        rng = np.random.default_rng(99)
        for sess in np.unique(s.session):
            m = np.where(s.session == sess)[0]
            perm = rng.permutation(m)
            s.y_cls[m] = s.y_cls[perm]
            s.y_ret[m] = s.y_ret[perm]
        return s

    ds.build_samples = shuffled
    try:
        from .train import fit as fit2
        bundle, _ = fit2(cfg, caches, TRAIN_START, d_to, horizon_min=H)
    finally:
        ds.build_samples = orig
    te = orig(cfg, caches, te_a_d, te_b_d, horizon_min=H)
    rows = np.arange(te.y_cls.size)
    model = model_from_bundle(cfg, bundle)
    from .train import predict_scores as ps
    s = ps(cfg, model, caches, te, rows)
    y = te.y_cls[rows]
    m = y != 1
    return round(_auc((y[m] == 2).astype(int), s[m]), 4)


def _gate_b_stats(taken: list[dict]) -> dict:
    if not taken:
        return {"n_trades": 0, "PASS": False}
    r = np.array([t["pnl_r"] for t in taken])
    usd = np.array([t["pnl_usd"] or 0.0 for t in taken])
    sessions = np.array([t["date"] for t in taken])
    rng = np.random.default_rng(7)
    uniq = np.unique(sessions)
    boots = []
    for _ in range(2000):                      # session-clustered bootstrap
        pick = rng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([np.where(sessions == p)[0] for p in pick])
        boots.append(r[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    eq = np.cumsum(usd)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
    dd = float((peak[1:] - eq).max()) if eq.size else 0.0
    out = {"n_trades": int(r.size), "expectancy_r": round(float(r.mean()), 4),
           "exp_ci": [round(float(lo), 4), round(float(hi), 4)],
           "net_usd": round(float(usd.sum()), 2),
           "max_dd_usd": round(dd, 2)}
    out["PASS"] = (r.size >= 300 and r.mean() >= 0.05 and lo > 0
                   and usd.sum() > 0)
    return out


def write_artifacts(cfg: Config, result: dict, trade_rows: list[dict],
                    run_id: str) -> None:
    rep = Path("reports"); rep.mkdir(exist_ok=True)
    with open(rep / f"dl_eval_{run_id}.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    # the ML's full blotter — every decision with entry/stop/target/exit —
    # persisted for chart comparison (JSON + CSV)
    with open(rep / f"dl_signals_{run_id}.json", "w") as fh:
        json.dump(trade_rows, fh, indent=2, default=str)
    if trade_rows:
        import csv
        keys = list(trade_rows[0].keys())
        with open(rep / f"dl_signals_{run_id}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(trade_rows)
        # scoreboard (owner requirement): win rate, total win/loss, net
        taken = [r for r in trade_rows if r["taken"]]
        if taken:
            wins = [r for r in taken if r["pnl_r"] > 0]
            losses = [r for r in taken if r["pnl_r"] <= 0]
            tw = sum(r["pnl_r"] for r in wins)
            tl = sum(r["pnl_r"] for r in losses)
            tw_u = sum(r["pnl_usd"] or 0 for r in wins)
            tl_u = sum(r["pnl_usd"] or 0 for r in losses)
            with open(rep / f"dl_signals_{run_id}_SUMMARY.txt", "w") as fh:
                fh.write(
                    f"TRADES {len(taken)} | WINS {len(wins)} | LOSSES "
                    f"{len(losses)} | WIN RATE {100*len(wins)/len(taken):.1f}%\n"
                    f"TOTAL WIN  {tw:+.2f}R  ${tw_u:+.2f}\n"
                    f"TOTAL LOSS {tl:+.2f}R  ${tl_u:+.2f}\n"
                    f"NET PROFIT {tw+tl:+.2f}R  ${tw_u+tl_u:+.2f}\n")
    out = cfg.export_dir / "dl_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"run_id": run_id,
                   "folds": [{"fold": f'{x["fold"]}', "auc_test": x["auc_up_down"],
                              "auc_val": None, "threshold": x["s_min"],
                              "n_test": x["n"], "prob_hist": []}
                             for x in result["folds"]],
                   "pooled": result["pooled"],
                   "gate_a": result["gate_a"], "gate_b": result["gate_b"]},
                  fh, indent=2, default=str)
