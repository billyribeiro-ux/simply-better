"""The insight engine — autonomous market-structure discovery in plain English.

The machine enumerates conditional hypotheses about its own tape (VWAP-stretch
reversion, prior-day level behavior, gap dynamics, opening-range/trend
structure, time-of-extremes), measures each one, and reports ONLY the facts
that survive its own statistical police:

- DISCOVERY on the first 70% of sessions, CONFIRMATION on the held-out 30% —
  a fact must show the same-direction lift in both, with adequate samples;
- Benjamini-Hochberg false-discovery-rate control across the whole hypothesis
  family (the machine corrects itself for multiple testing — the "look at
  enough patterns and some lie" trap);
- every statistic is computed point-in-time (running VWAP, prior-day levels
  known at the open, daily context from PRIOR days only).

Output: dashboard/static/data/market_insights.json — plain-English statements
with their numbers (frequency, base rate, sample sizes, confirmation stats).
Runs nightly inside `mie daily`; the fact base updates as the tape grows.

These are DESCRIPTIVE market facts with honest uncertainty — the knowledge
layer. They are not auto-traded; anything promoted to a trading rule still
goes through the ledger's pre-registration discipline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config
from .data import Store

log = logging.getLogger("engine.discover")

TRAIN_FRAC = 0.7
FDR_Q = 0.05
MIN_TRAIN, MIN_TEST = 40, 20


# ---------------------------------------------------------------------------
# per-session point-in-time record
# ---------------------------------------------------------------------------
@dataclass
class Session:
    symbol: str
    day: date
    o: np.ndarray; h: np.ndarray; l: np.ndarray; c: np.ndarray; v: np.ndarray
    vwap: np.ndarray            # running, causal
    atr: float                  # prior-day-based daily ATR
    prior_high: float; prior_low: float; prior_close: float
    above_sma20: bool           # daily-chart confluence (prior closes only)
    prior_day_up: bool


def _sessions(cfg: Config) -> list[Session]:
    store = Store(cfg)
    far, today = date(2000, 1, 1), date(2100, 1, 1)
    out: list[Session] = []
    for sym in cfg.universe:
        b5 = store.load_bars("5min", sym, far, today).sort("ts")
        daily = store.load_daily(sym).sort("date")
        b5 = b5.with_columns(pl.col("ts").dt.date().alias("dt"))
        # trailing ATR(14) and SMA(20) from PRIOR days only
        dcl = daily["close"].to_numpy().astype(float)
        dhi = daily["high"].to_numpy().astype(float)
        dlo = daily["low"].to_numpy().astype(float)
        ddate = daily["date"].to_list()             # python dates (group keys)
        tr = np.maximum(dhi[1:] - dlo[1:],
                        np.maximum(abs(dhi[1:] - dcl[:-1]), abs(dlo[1:] - dcl[:-1])))
        atr14 = np.full(len(dcl), np.nan)
        for i in range(15, len(dcl)):
            atr14[i] = tr[i-15:i-1].mean()          # prior days only
        sma20 = np.full(len(dcl), np.nan)
        for i in range(20, len(dcl)):
            sma20[i] = dcl[i-20:i].mean()           # prior closes only
        didx = {d: i for i, d in enumerate(ddate)}

        for (day,), g in b5.group_by(["dt"], maintain_order=True):
            if g.height < 60:
                continue
            i = didx.get(day)
            if i is None or i < 21 or not np.isfinite(atr14[i]):
                continue
            o = g["open"].to_numpy(); h = g["high"].to_numpy()
            l = g["low"].to_numpy(); c = g["close"].to_numpy()
            v = g["volume"].to_numpy()
            tp = (h + l + c) / 3.0
            cv = np.cumsum(v)
            vwap = np.cumsum(tp * v) / np.where(cv > 0, cv, 1)
            out.append(Session(
                symbol=sym, day=day, o=o, h=h, l=l, c=c, v=v, vwap=vwap,
                atr=float(atr14[i]),
                prior_high=float(dhi[i-1]), prior_low=float(dlo[i-1]),
                prior_close=float(dcl[i-1]),
                above_sma20=bool(dcl[i-1] > sma20[i]),
                prior_day_up=bool(dcl[i-1] > dcl[i-2]),
            ))
    out.sort(key=lambda s: s.day)
    return out


# ---------------------------------------------------------------------------
# hypothesis families — each yields (key, english, event_fn, outcome_fn)
# event_fn(sess) -> index of first trigger bar or None (point-in-time)
# outcome_fn(sess, i) -> bool
# ---------------------------------------------------------------------------
def _first_stretch(sess: Session, k: float, side: int):
    """First bar whose close is >= k*ATR above (side=+1) / below (-1) VWAP,
    after bar 6 and before the last 12 bars (room for the outcome window)."""
    d = side * (sess.c - sess.vwap)
    for i in range(6, len(sess.c) - 12):
        if d[i] >= k * sess.atr:
            return i
    return None


def _reverts_half(sess: Session, i: int, side: int, within: int = 12) -> bool:
    """Price comes at least halfway back to VWAP within `within` bars."""
    target = sess.vwap[i] + side * 0.5 * (sess.c[i] - sess.vwap[i])
    for j in range(i + 1, min(i + 1 + within, len(sess.c))):
        if side > 0 and sess.l[j] <= target:
            return True
        if side < 0 and sess.h[j] >= target:
            return True
    return False


def _hypotheses():
    H = []
    for k in (1.0, 1.5, 2.0):
        for side, word in ((1, "above"), (-1, "below")):
            H.append((
                f"vwap_stretch_{word}_{k}",
                f"stretched >= {k} ATR {word} VWAP: price retraces at least "
                f"halfway back to VWAP within the next hour",
                lambda s, k=k, side=side: _first_stretch(s, k, side),
                lambda s, i, side=side: _reverts_half(s, i, side),
            ))

    def _touch_pdh(s: Session):
        for i in range(3, len(s.c) - 12):
            if s.h[i] >= s.prior_high:
                return i
        return None

    def _touch_pdl(s: Session):
        for i in range(3, len(s.c) - 12):
            if s.l[i] <= s.prior_low:
                return i
        return None

    H.append(("pdh_breaks_through",
              "touches yesterday's HIGH: closes the day above it",
              _touch_pdh, lambda s, i: s.c[-1] > s.prior_high))
    H.append(("pdh_rejects",
              "touches yesterday's HIGH: rejects >= 0.3 ATR within the hour",
              _touch_pdh,
              lambda s, i: min(s.l[i+1:i+13]) <= s.prior_high - 0.3*s.atr
              if i + 1 < len(s.l) else False))
    H.append(("pdl_breaks_through",
              "touches yesterday's LOW: closes the day below it",
              _touch_pdl, lambda s, i: s.c[-1] < s.prior_low))
    H.append(("pdl_rejects",
              "touches yesterday's LOW: bounces >= 0.3 ATR within the hour",
              _touch_pdl,
              lambda s, i: max(s.h[i+1:i+13]) >= s.prior_low + 0.3*s.atr
              if i + 1 < len(s.h) else False))

    def _gap(s: Session, gmin: float, up: bool):
        gap = (s.o[0] - s.prior_close) / s.atr
        ok = gap >= gmin if up else gap <= -gmin
        return 0 if ok else None

    for gmin in (0.3, 0.75):
        H.append((f"gap_up_{gmin}_fills",
                  f"gaps UP >= {gmin} ATR at the open: trades back to "
                  f"yesterday's close the same day (gap fill)",
                  lambda s, gmin=gmin: _gap(s, gmin, True),
                  lambda s, i: min(s.l) <= s.prior_close))
        H.append((f"gap_down_{gmin}_fills",
                  f"gaps DOWN >= {gmin} ATR at the open: trades back up to "
                  f"yesterday's close the same day (gap fill)",
                  lambda s, gmin=gmin: _gap(s, gmin, False),
                  lambda s, i: max(s.h) >= s.prior_close))

    def _first_hour_up(s: Session):
        return 12 if s.c[11] > s.o[0] else None

    def _first_hour_down(s: Session):
        return 12 if s.c[11] < s.o[0] else None

    H.append(("fh_up_lod_in",
              "first hour closes UP: the day's LOW is already in place",
              _first_hour_up, lambda s, i: int(np.argmin(s.l)) < 12))
    H.append(("fh_down_hod_in",
              "first hour closes DOWN: the day's HIGH is already in place",
              _first_hour_down, lambda s, i: int(np.argmax(s.h)) < 12))
    return H


CONDITIONS = [
    ("all days", lambda s: True),
    ("daily uptrend (above 20-day average)", lambda s: s.above_sma20),
    ("daily downtrend (below 20-day average)", lambda s: not s.above_sma20),
]


def _bh_fdr(pvals: list[float], q: float) -> list[bool]:
    n = len(pvals)
    order = np.argsort(pvals)
    keep = [False] * n
    thresh = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / n:
            thresh = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            keep[idx] = True
    return keep


def _binom_p(hits: int, n: int, p0: float) -> float:
    """Two-sided exact-ish binomial p-value vs base rate p0 (normal approx
    with continuity correction; adequate at n>=40)."""
    if n == 0:
        return 1.0
    sd = np.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    z = (abs(hits - n * p0) - 0.5) / sd
    from math import erfc, sqrt
    return erfc(max(z, 0.0) / sqrt(2))


def run(cfg: Config) -> dict:
    sessions = _sessions(cfg)
    days = sorted({s.day for s in sessions})
    cut = days[int(len(days) * TRAIN_FRAC)]
    train = [s for s in sessions if s.day <= cut]
    test = [s for s in sessions if s.day > cut]
    log.info("discovery corpus: %d sessions train / %d test (cut %s)",
             len(train), len(test), cut)

    hyps = _hypotheses()
    symbols = ["ALL"] + list(cfg.universe)
    cands = []
    for key, english, ev, oc in hyps:
        for cond_name, cond in CONDITIONS:
            for sym in symbols:
                def stats(pool):
                    hits = tot = 0
                    for s in pool:
                        if sym != "ALL" and s.symbol != sym:
                            continue
                        if not cond(s):
                            continue
                        i = ev(s)
                        if i is None:
                            continue
                        tot += 1
                        hits += bool(oc(s, i))
                    return hits, tot
                h_tr, n_tr = stats(train)
                if n_tr < MIN_TRAIN:
                    continue
                h_te, n_te = stats(test)
                if n_te < MIN_TEST:
                    continue
                cands.append({
                    "key": key, "english": english, "cond": cond_name,
                    "symbol": sym,
                    "train_pct": 100*h_tr/n_tr, "train_n": n_tr,
                    "test_pct": 100*h_te/n_te, "test_n": n_te,
                    "p": _binom_p(h_tr, n_tr, 0.5),
                })
    # FDR police on the DISCOVERY sample, confirmation on the held-out sample
    keep = _bh_fdr([c["p"] for c in cands], FDR_Q)
    facts = []
    for c, k in zip(cands, keep):
        if not k:
            continue
        lift_tr = c["train_pct"] - 50.0
        lift_te = c["test_pct"] - 50.0
        if lift_tr * lift_te <= 0:            # must confirm out-of-sample
            continue
        if abs(lift_te) < 5.0:                # confirmation must be material
            continue
        subject = c["symbol"] if c["symbol"] != "ALL" else "these 8 names"
        pct = c["test_pct"]
        facts.append({
            "statement": (
                f"When {subject} {c['english']} — it happens "
                f"{pct:.0f}% of the time"
                + ("" if c["cond"] == "all days" else f" ({c['cond']})")),
            "symbol": c["symbol"], "condition": c["cond"], "family": c["key"],
            "discovered_pct": round(c["train_pct"], 1),
            "discovered_n": c["train_n"],
            "confirmed_pct": round(c["test_pct"], 1),
            "confirmed_n": c["test_n"],
        })
    facts.sort(key=lambda f: -abs(f["confirmed_pct"] - 50))
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": len(sessions),
        "hypotheses_tested": len(cands),
        "facts_confirmed": len(facts),
        "method": (f"discover on first {int(TRAIN_FRAC*100)}% of sessions, "
                   f"confirm on held-out {int((1-TRAIN_FRAC)*100)}%, "
                   f"Benjamini-Hochberg FDR q={FDR_Q}"),
        "facts": facts,
    }
    log.info("insights: %d/%d hypotheses survived discovery+confirmation",
             len(facts), len(cands))
    return out


def write(cfg: Config, insights: dict) -> Path:
    out = cfg.export_dir / "market_insights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(insights, fh, indent=2)
    return out
