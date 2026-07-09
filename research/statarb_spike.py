"""Cross-sectional intraday residual mean-reversion — edge spike.

NO LOOKAHEAD: per-name 2-factor (SPY,QQQ) betas and residual vol are estimated
on TRAILING sessions only (strictly before the trading session); everything at
bar t uses data <= t. Measures whether an extreme intraday residual s-score
predicts reversion, gross and net of measured per-symbol cost.
"""
import numpy as np, polars as pl

FACTORS = ["SPY", "QQQ"]
NAMES = ["AAPL", "NVDA", "AMZN", "NFLX", "TSLA"]     # idiosyncratic book
ALL = FACTORS + NAMES
# measured per-side slippage bps (Corwin-Schultz, from the engine config)
SLIP = {"SPY":0.7,"QQQ":0.9,"IWM":0.9,"AAPL":1.3,"AMZN":1.5,"NFLX":1.4,"NVDA":2.3,"TSLA":2.9}
COMM_BPS = lambda px: 0.0035/px*1e4          # commission in bps of notional

TRAIL = 20          # trailing sessions for beta + resid vol
K = 6               # forward horizon (bars) = 30 min
S_ENTRY = 1.5       # s-score entry threshold
MIN_BARS = 6        # only trade after >=6 bars since open (>=10:00)

# ---- load aligned panel of 5-min log returns -----------------------------
frames = []
for s in ALL:
    d = (pl.read_parquet(f"data/parquet/5min/{s}.parquet")
         .select(["ts", "close"]).rename({"close": s}))
    frames.append(d)
panel = frames[0]
for d in frames[1:]:
    panel = panel.join(d, on="ts", how="inner")
panel = panel.sort("ts")
panel = panel.with_columns(pl.col("ts").dt.date().alias("date"))
ts = panel["ts"].to_numpy()
dates = panel["date"].to_numpy()
px = {s: panel[s].to_numpy() for s in ALL}
ret = {s: np.concatenate([[0.0], np.diff(np.log(px[s]))]) for s in ALL}
# zero the first bar of each session (overnight gap is not an intraday return)
sess_ids = panel["date"].to_numpy()
new_sess = np.concatenate([[True], sess_ids[1:] != sess_ids[:-1]])
for s in ALL:
    ret[s][new_sess] = 0.0

uniq_dates = list(dict.fromkeys(dates.tolist()))
date_to_rows = {d: np.where(dates == d)[0] for d in uniq_dates}

F = np.column_stack([ret[f] for f in FACTORS])   # (T,2)

all_s, all_fwd, all_name = [], [], []
strat_pnl_gross, strat_pnl_net, strat_names = [], [], []

for di in range(TRAIL, len(uniq_dates)):
    d = uniq_dates[di]
    rows = date_to_rows[d]
    if len(rows) < MIN_BARS + K + 1:
        continue
    # trailing window rows (strictly earlier sessions)
    tr_dates = uniq_dates[di-TRAIL:di]
    tr_rows = np.concatenate([date_to_rows[x] for x in tr_dates])
    for nm in NAMES:
        y_tr = ret[nm][tr_rows]
        X_tr = F[tr_rows]
        # OLS beta on trailing bars (add intercept)
        A = np.column_stack([np.ones(len(tr_rows)), X_tr])
        beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
        # trailing residual vol (per 5-min bar)
        resid_tr = y_tr - A @ beta
        sig = float(np.std(resid_tr))
        if sig <= 0:
            continue
        # intraday residuals for session d
        rr = rows
        yr = ret[nm][rr]
        Ar = np.column_stack([np.ones(len(rr)), F[rr]])
        resid = yr - Ar @ beta
        Xcum = np.cumsum(resid)                    # cumulative residual since open
        nbar = np.arange(1, len(rr) + 1)
        sscore = Xcum / (sig * np.sqrt(nbar))      # random-walk normalization
        # observations: entries with enough room and forward horizon
        for j in range(MIN_BARS, len(rr) - K):
            s_now = sscore[j]
            fwd = Xcum[j + K] - Xcum[j]             # forward residual return
            all_s.append(s_now); all_fwd.append(fwd); all_name.append(nm)
            if abs(s_now) >= S_ENTRY:
                pnl = -np.sign(s_now) * fwd        # fade the dislocation
                # cost: round-trip on the NAME leg + hedge legs (SPY,QQQ)
                px_j = px[nm][rr[j]]
                name_cost = 2*(SLIP[nm] + COMM_BPS(px_j))/1e4
                hedge_cost = sum(abs(beta[1+k])*2*(SLIP[f])/1e4
                                 for k, f in enumerate(FACTORS))
                strat_pnl_gross.append(pnl)
                strat_pnl_net.append(pnl - name_cost - hedge_cost)
                strat_names.append(nm)

s = np.array(all_s); fwd = np.array(all_fwd)
print(f"observations: {len(s)}  (names={NAMES}, factors={FACTORS})")
# reversion signature: Spearman(s, forward residual) should be NEGATIVE
from scipy.stats import spearmanr
rho, p = spearmanr(s, fwd)
print(f"Spearman(s-score, forward {K*5}min residual) = {rho:+.4f}  (p={p:.2e})")
print("  reversion => negative; momentum => positive")
# bucketed forward residual by s-score decile
order = np.argsort(s)
print("\ns-score decile | mean s | mean fwd residual (bps)")
for q in range(10):
    idx = order[q*len(s)//10:(q+1)*len(s)//10]
    print(f"  d{q}: s={s[idx].mean():+.2f}  fwd={fwd[idx].mean()*1e4:+.2f} bps  n={len(idx)}")

g = np.array(strat_pnl_gross); nt = np.array(strat_pnl_net)
print(f"\nreversion strategy (|s|>={S_ENTRY}, hold {K*5}min, fade):")
print(f"  trades={len(g)}")
print(f"  GROSS: mean {g.mean()*1e4:+.2f} bps/trade  total {g.sum()*1e4:+.0f} bps")
print(f"  NET  : mean {nt.mean()*1e4:+.2f} bps/trade  total {nt.sum()*1e4:+.0f} bps")
# bootstrap CI on net per-trade
rng = np.random.default_rng(7)
b = np.array([rng.choice(nt, nt.size, replace=True).mean() for _ in range(4000)])
print(f"  NET per-trade 95% CI: [{np.percentile(b,2.5)*1e4:+.2f}, {np.percentile(b,97.5)*1e4:+.2f}] bps")
# per-name net
print("  per-name net (bps/trade):")
for nm in NAMES:
    m = np.array(strat_names) == nm
    if m.sum(): print(f"    {nm}: {nt[m].mean()*1e4:+.2f} bps  (n={m.sum()})")

# ---- horizon + construction robustness (confirm the null, don't fish) -----
print("\n=== horizon sweep: Spearman(s-score, forward residual) ===")
# rebuild observations per horizon from stored (need per-(i,session) arrays) -
# quick recompute over the same loop but multi-K
for Kx in [1, 2, 3, 6, 12, 24]:
    S_, FWD_ = [], []
    for di in range(TRAIL, len(uniq_dates)):
        d = uniq_dates[di]; rows = date_to_rows[d]
        if len(rows) < MIN_BARS + Kx + 1: continue
        tr_dates = uniq_dates[di-TRAIL:di]
        tr_rows = np.concatenate([date_to_rows[x] for x in tr_dates])
        for nm in NAMES:
            y_tr = ret[nm][tr_rows]; X_tr = F[tr_rows]
            A = np.column_stack([np.ones(len(tr_rows)), X_tr])
            beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
            sig = float(np.std(y_tr - A @ beta))
            if sig <= 0: continue
            rr = rows; Ar = np.column_stack([np.ones(len(rr)), F[rr]])
            resid = ret[nm][rr] - Ar @ beta
            Xcum = np.cumsum(resid); nbar = np.arange(1, len(rr)+1)
            ss = Xcum/(sig*np.sqrt(nbar))
            for j in range(MIN_BARS, len(rr)-Kx):
                S_.append(ss[j]); FWD_.append(Xcum[j+Kx]-Xcum[j])
    S_ = np.array(S_); FWD_ = np.array(FWD_)
    rho,_ = spearmanr(S_, FWD_)
    print(f"  K={Kx:2d} ({Kx*5:3d}min): Spearman={rho:+.4f}  n={len(S_)}")
