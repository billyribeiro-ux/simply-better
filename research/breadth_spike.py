"""Pre-registered BREADTH spike (ledger 2026-07-10): the 8-name construction
of research/statarb_spike.py, unchanged (TRAIL=20, K=6, |s|>=1.5, MIN_BARS=6,
SPY/QQQ 2-factor betas), applied to the 49-single-name cross-section with
per-name Corwin-Schultz-measured costs. Criteria (a)/(b)/(c) fixed a priori."""
import numpy as np, polars as pl
from scipy.stats import spearmanr

FACTORS = ["SPY", "QQQ"]
NAMES = ["AAPL","NVDA","AMZN","NFLX","TSLA","MSFT","GOOGL","META","AVGO","AMD",
         "CRM","ORCL","ADBE","INTC","QCOM","TXN","MU","JPM","BAC","WFC","GS",
         "MS","SCHW","C","LLY","UNH","JNJ","PFE","MRK","ABBV","TMO","XOM",
         "CVX","COP","SLB","HD","WMT","COST","MCD","NKE","SBUX","TGT","BA",
         "CAT","GE","UNP","HON","DE","DIS","UBER"]
ALL = FACTORS + NAMES
TRAIL, K, S_ENTRY, MIN_BARS = 20, 6, 1.5, 6
COMM = 0.0035

# ---- per-name Corwin-Schultz half-spread (same estimator as the 8) --------
kcs = 3 - 2*np.sqrt(2)
def cs_bps_per_side(h, l):
    hi2=np.maximum(h[:-1],h[1:]); lo2=np.minimum(l[:-1],l[1:])
    b=(np.log(h[:-1]/l[:-1]))**2+(np.log(h[1:]/l[1:]))**2
    g=(np.log(hi2/lo2))**2
    a=(np.sqrt(2*b)-np.sqrt(b))/kcs - np.sqrt(g/kcs)
    s=2*(np.exp(a)-1)/(1+np.exp(a)); s=s[np.isfinite(s)]; s=np.clip(s,0,None)
    return float(np.median(s))*1e4/2

SLIP = {}
frames = []
for sym in ALL:
    d = pl.read_parquet(f"data/parquet/5min/{sym}.parquet").sort("ts")
    h = d["high"].to_numpy(); l = d["low"].to_numpy()
    SLIP[sym] = cs_bps_per_side(h, l)
    frames.append(d.select(["ts","close"]).rename({"close": sym}))
panel = frames[0]
for d in frames[1:]:
    panel = panel.join(d, on="ts", how="inner")
panel = panel.sort("ts").with_columns(pl.col("ts").dt.date().alias("date"))
dates = panel["date"].to_numpy()
px = {s: panel[s].to_numpy() for s in ALL}
ret = {s: np.concatenate([[0.0], np.diff(np.log(px[s]))]) for s in ALL}
new = np.concatenate([[True], dates[1:] != dates[:-1]])
for s in ALL: ret[s][new] = 0.0
uniq = list(dict.fromkeys(dates.tolist()))
d2r = {d: np.where(dates == d)[0] for d in uniq}
F = np.column_stack([ret[f] for f in FACTORS])

S_, FWD_, SESS_, NM_ = [], [], [], []
pnl_net, pnl_gross, pnl_nm, pnl_sess = [], [], [], []
for di in range(TRAIL, len(uniq)):
    d = uniq[di]; rows = d2r[d]
    if len(rows) < MIN_BARS + K + 1: continue
    tr = np.concatenate([d2r[x] for x in uniq[di-TRAIL:di]])
    A = np.column_stack([np.ones(len(tr)), F[tr]])
    Ar = np.column_stack([np.ones(len(rows)), F[rows]])
    for nm in NAMES:
        beta, *_ = np.linalg.lstsq(A, ret[nm][tr], rcond=None)
        sig = float(np.std(ret[nm][tr] - A @ beta))
        if sig <= 0: continue
        resid = ret[nm][rows] - Ar @ beta
        X = np.cumsum(resid); nb = np.arange(1, len(rows)+1)
        ss = X/(sig*np.sqrt(nb))
        for j in range(MIN_BARS, len(rows)-K):
            S_.append(ss[j]); FWD_.append(X[j+K]-X[j]); SESS_.append(str(d)); NM_.append(nm)
            if abs(ss[j]) >= S_ENTRY:
                p = -np.sign(ss[j])*(X[j+K]-X[j])
                name_c = 2*(SLIP[nm] + COMM/px[nm][rows[j]]*1e4)/1e4
                hedge_c = sum(abs(beta[1+k])*2*SLIP[f]/1e4 for k,f in enumerate(FACTORS))
                pnl_gross.append(p); pnl_net.append(p-name_c-hedge_c)
                pnl_nm.append(nm); pnl_sess.append(str(d))

S_=np.array(S_); FWD_=np.array(FWD_); SESS_=np.array(SESS_)
rho, pv = spearmanr(S_, FWD_)
# session-clustered t on per-session Spearman
per=[]
for s in np.unique(SESS_):
    m = SESS_==s
    if m.sum()>=200:
        ic = spearmanr(S_[m], FWD_[m]).statistic
        if np.isfinite(ic): per.append(ic)
per=np.array(per)
t = per.mean()/(per.std(ddof=1)/np.sqrt(per.size))
print(f"observations: {len(S_)} across {len(NAMES)} names, {per.size} sessions")
print(f"pooled Spearman(s, fwd 30min residual) = {rho:+.4f} (p={pv:.1e})")
print(f"session-clustered: mean per-session IC = {per.mean():+.4f}, t = {t:+.2f}")
g=np.array(pnl_gross); n_=np.array(pnl_net)
print(f"\nfrozen fade rule: trades={len(g)}")
print(f"  GROSS {g.mean()*1e4:+.2f} bps/trade | NET {n_.mean()*1e4:+.2f} bps/trade | total net {n_.sum()*1e4:+.0f} bps")
rng=np.random.default_rng(7)
sess_arr=np.array(pnl_sess); uq=np.unique(sess_arr)
boots=[]
for _ in range(2000):
    pick=rng.choice(uq,uq.size,replace=True)
    idx=np.concatenate([np.where(sess_arr==p)[0] for p in pick])
    boots.append(n_[idx].mean())
lo,hi=np.percentile(boots,[2.5,97.5])
print(f"  NET/trade 95% session-clustered CI: [{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps")
print(f"\nCRITERIA: (a) rho<=-0.015 & |t|>=4 & net>0 | (c) |rho|<0.010 or wrong sign")
print(f"  rho={rho:+.4f}  t={t:+.2f}  net_mean={n_.mean()*1e4:+.2f}bps -> ", end="")
if rho <= -0.015 and abs(t) >= 4 and n_.mean() > 0: print("OUTCOME (a) BUILD")
elif abs(rho) < 0.010 or rho > 0: print("OUTCOME (c) DEAD — stop intraday cross-sectional research")
else: print("OUTCOME (b) bench")
