"""The self-learning loop's brain: figure out WHY losses happen, then act.

Mechanism:
1. On the accumulated history of TAKEN trades (all folds decided so far), fit
   a shallow decision tree predicting loss vs win over the feature space.
2. Every leaf whose loss rate exceeds the trigger with enough support becomes
   an AdjustmentRule — a conjunction of feature conditions plus a probability
   bump: trades landing in that regime must clear a higher bar.
3. Rules learned through fold k are applied only to fold k+1 onward, so every
   adjustment is evaluated strictly out-of-sample.
4. KMeans clustering of losing trades produces the human-readable diagnosis
   surfaced on the dashboard ("losses concentrate in ...").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from .config import Config
from .features import FEATURES

log = logging.getLogger("engine.attribution")

_LABELS = {
    "minutes_since_open": "min since open",
    "gap_pct": "gap %",
    "range_ext_atr": "range extension (ATR)",
    "vol_z": "volume z",
    "dist_vwap_atr": "dist from VWAP (ATR)",
    "prior_high_dist_atr": "dist to prior high (ATR)",
    "prior_low_dist_atr": "dist to prior low (ATR)",
    "gk_vol_z": "GK vol z",
    "ret_30m_atr": "30m momentum (ATR)",
    "wick_ratio": "wick ratio",
    "n_extremes": "new-extreme count",
    "dow": "day of week",
    "atr_pct": "ATR %",
    "tod_sin": "time-of-day (sin)",
    "tod_cos": "time-of-day (cos)",
    "side": "side",
}


@dataclass
class Condition:
    feature: str
    op: str          # "<=" | ">"
    value: float

    def match(self, row: dict) -> bool:
        v = float(row[self.feature])
        return v <= self.value if self.op == "<=" else v > self.value

    def text(self) -> str:
        return f"{_LABELS.get(self.feature, self.feature)} {self.op} {self.value:.2f}"


@dataclass
class AdjustmentRule:
    conditions: list[Condition]
    loss_rate: float
    support: int
    bump: float
    learned_after_fold: str

    def match(self, row: dict) -> bool:
        return all(c.match(row) for c in self.conditions)

    def text(self) -> str:
        return " AND ".join(c.text() for c in self.conditions)


@dataclass
class Attribution:
    rules: list[AdjustmentRule] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)

    def effective_threshold(self, row: dict, base: float, fold: str) -> float:
        thr = base
        for r in self.rules:
            if r.learned_after_fold < fold and r.match(row):
                thr += r.bump
        return min(thr, 0.95)

    def rules_json(self) -> list[dict]:
        return [
            {"text": r.text(), "loss_rate": round(r.loss_rate, 3),
             "support": r.support, "bump": r.bump,
             "learned_after_fold": r.learned_after_fold}
            for r in self.rules
        ]


def learn_rules(cfg: Config, taken: pl.DataFrame, current_fold: str,
                attribution: Attribution) -> None:
    """Mine loss regimes from taken-trade history and append new rules."""
    lcfg = cfg.learning
    if taken.height < int(lcfg["min_leaf_support"]) * 2:
        return
    X = taken.select(FEATURES).to_numpy()
    y_loss = (taken["pnl_r"].to_numpy() <= 0).astype(int)
    if y_loss.sum() < 5 or y_loss.sum() == len(y_loss):
        return

    tree = DecisionTreeClassifier(
        max_depth=int(lcfg["tree_depth"]),
        min_samples_leaf=int(lcfg["min_leaf_support"]),
        random_state=7,
    )
    tree.fit(X, y_loss)

    t = tree.tree_
    existing = {r.text() for r in attribution.rules}

    def walk(node: int, conds: list[Condition]) -> None:
        if t.children_left[node] == -1:                     # leaf
            n = int(t.n_node_samples[node])
            losses = float(t.value[node][0][1])
            loss_rate = losses / max(n, 1)
            if (n >= int(lcfg["min_leaf_support"])
                    and loss_rate > float(lcfg["loss_rate_trigger"])
                    and conds):
                rule = AdjustmentRule(
                    conditions=list(conds), loss_rate=loss_rate, support=n,
                    bump=float(lcfg["threshold_bump"]),
                    learned_after_fold=current_fold,
                )
                if rule.text() not in existing:
                    attribution.rules.append(rule)
                    existing.add(rule.text())
                    log.info("new rule after %s: [%s] loss=%.0f%% n=%d",
                             current_fold, rule.text(), loss_rate * 100, n)
            return
        f = FEATURES[t.feature[node]]
        thr = float(t.threshold[node])
        walk(t.children_left[node], conds + [Condition(f, "<=", thr)])
        walk(t.children_right[node], conds + [Condition(f, ">", thr)])

    walk(0, [])


def cluster_losses(taken: pl.DataFrame) -> list[dict]:
    """KMeans diagnosis of where losses live, for the dashboard."""
    losses = taken.filter(pl.col("pnl_r") <= 0)
    wins = taken.filter(pl.col("pnl_r") > 0)
    if losses.height < 12 or wins.height < 12:
        return []
    Xl = losses.select(FEATURES).to_numpy()
    mu = Xl.mean(axis=0)
    sd = Xl.std(axis=0)
    sd[sd == 0] = 1.0
    k = min(3, max(1, losses.height // 15))
    km = KMeans(n_clusters=k, n_init=10, random_state=7)
    lab = km.fit_predict((Xl - mu) / sd)
    win_mu = wins.select(FEATURES).to_numpy().mean(axis=0)

    out: list[dict] = []
    for ci in range(k):
        mask = lab == ci
        cent = Xl[mask].mean(axis=0)
        delta = (cent - win_mu) / sd
        top = np.argsort(-np.abs(delta))[:3]
        drivers = [
            {"feature": _LABELS.get(FEATURES[i], FEATURES[i]),
             "delta": round(float(delta[i]), 2),
             "loss_mean": round(float(cent[i]), 2),
             "win_mean": round(float(win_mu[i]), 2)}
            for i in top
        ]
        avg_r = float(losses.filter(pl.Series(mask))["pnl_r"].mean())
        out.append({"label": f"Loss cluster {ci + 1}", "size": int(mask.sum()),
                    "avg_pnl_r": round(avg_r, 2), "drivers": drivers})
    return out
