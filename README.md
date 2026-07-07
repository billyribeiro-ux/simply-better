# MIE HOD/LOD Self-Learning Engine

A self-learning intraday ML engine for two setups on an 8-name liquid universe
(AAPL, NVDA, AMZN, NFLX, SPY, QQQ, IWM, TSLA):

- **HOD_FADE** — short a rejection at a fresh high-of-day
- **LOD_RECLAIM** — long a reclaim at a fresh low-of-day

Data comes from FMP (Financial Modeling Prep). Detection runs on 5-minute
bars; entries are confirmed and simulated on 1-minute bars. Results are
served to a static SvelteKit research dashboard.

## What the machine learns

**Which triggers to take.** A LightGBM meta-model scores every detected
trigger. This is meta-labeling (López de Prado): the label is "did this trade
reach its target before its stop," not "did price go up," so the model learns
which primary signals are worth trading rather than trying to predict raw
direction. The decision threshold is optimized on the chronological tail of
each training window — never on the fold being predicted.

**Trade geometry.** Stop and target distances per (setup × vol-regime) cell
are learned from the empirical MAE/MFE first-crossing distributions of
training events only, choosing the pair that maximizes expectancy in R.
Position sizing is constant-risk: `shares = (equity × risk%) / stop distance`,
so a wide-stop trade risks the same dollars as a tight-stop trade.

**Why it loses.** A shallow decision tree mines loss regimes from
taken-trade history; each toxic leaf becomes a threshold-bump rule applied
only to strictly later folds. KMeans clustering of losing trades produces
the human-readable loss diagnosis on the dashboard.

## The institutional layer

Selection quality is protected by three mechanisms beyond the walk-forward
itself. **Out-of-fold calibration:** the isotonic calibrator and the decision
threshold learn from out-of-fold probabilities (expanding chronological
blocks inside the training window, purged with the same embargo as the
walk-forward) — never from the final model's memorized in-sample output.
**Seed-bagged ensemble:** each decision stack averages several LightGBM
members fitted with different seeds. **Uncertainty on every headline
number:** expectancy ships with a 95% trade-bootstrap CI and Sharpe with a
95% circular-block-bootstrap CI, so a point estimate can never pose as more
than the data supports. The portfolio layer adds a daily kill switch
(`execution.max_daily_loss_pct`): once a session's realized loss breaches
the limit, no new entries open that day.

## The validation discipline

Everything is walk-forward, embargoed, and out-of-sample. Daily context
features are shifted so an intraday session only sees prior-day data. Each
monthly fold's geometry, model, threshold, and adjustment rules are learned
strictly from data before that month (minus an embargo gap). Intrabar ties —
stop and target crossed in the same 1-minute bar — always resolve against
the trade. Slippage and commission are charged on both sides. Every run
computes the Probabilistic Sharpe Ratio and the Deflated Sharpe Ratio so the
backtest cannot hide overfitting behind a pretty equity curve.

## Known data constraint

FMP caps 1-minute history depth by plan tier (often ~1–3 months), while
5-minute reaches back years. Ingestion chunks requests (4 days per 1-min
call, 25 per 5-min call), but backtest depth on 1-minute confirmation is
bounded by your plan. If 1-minute coverage is missing for a range, entries
simply never confirm there and the pipeline raises with a clear message —
that is intended behavior, not a bug.

## Quickstart

```bash
# engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # or: uv pip install -e .
export FMP_API_KEY=...
mie ingest --from 2025-01-02 --to 2026-07-03
mie run    --from 2025-01-02 --to 2026-07-03   # trains, backtests, exports

# dashboard
cd dashboard && pnpm install && pnpm dev
```

`mie all --from … --to …` does ingest + run + export in one shot. The
dashboard ships with committed sample data (`run_id: "SAMPLE"`) so a fresh
clone renders immediately; a real `mie run` overwrites it.

## Backtest vs live

The walk-forward backtest is the validation device; the live path is its
production twin, built from the same code: same detector and features
(`setups.detect`), same fitting function (`model.fit_scored_model` — shared
byte-for-byte with every walk-forward fold), same geometry lookup, same
1-minute confirmation scan, same sizing arithmetic. Train/serve parity is
asserted by `tests/test_live_path.py`.

```bash
# 1. train the production model on everything cached (run ingest first)
mie train --from 2025-01-02 --to 2026-07-03

# 2. one-shot scan of today's session (US/Eastern)
mie live

# 3. keep scanning every 60s until the 15:55 ET flat-by
mie live --poll 60

# 4. replay any cached past session (dry run / verification)
mie live --date 2026-06-25 --no-refresh
```

Each scan refreshes FMP data (unless `--no-refresh`), prints the signal
table, writes `dashboard/static/data/live.json` for the terminal's Live
panel, and appends to the DuckDB `live_signals` record. Signals carry the
planned entry stop-order price (the 1-min break level), stop, target,
constant-risk share count (sized from `execution.starting_equity`), the
model's probability vs its effective threshold, and a status: `awaiting`
(inside the 15-bar confirmation window), `confirmed` (break traded, actual
entry shown), or `expired` (window passed without a break).

Live caveats, stated plainly: intraday data freshness is bounded by your
FMP plan tier (delayed feeds produce delayed signals); the detector needs
8 completed 5-minute bars, so the earliest trigger of a session surfaces a
few minutes after its bar closes (bounded latency, never lookahead); and
this is a **signal feed only — no broker connection, no order routing**.
Retrain (`mie train`) after each `mie ingest` refresh; the loader refuses a
model whose feature set no longer matches the code.

Suggested cron (ET session, weekdays):

```cron
35 9 * * 1-5  cd /path/to/repo && .venv/bin/mie live --poll 60
```

```bash
# verification
python3 -m py_compile src/engine/*.py   # compiles clean
python3 tests/smoke_synthetic.py        # synthetic end-to-end (writes data/, delete after)
pytest -q                               # engine math unit tests
cd dashboard && pnpm check && pnpm build
```

## How to read DSR and PSR honestly

**PSR** (Probabilistic Sharpe Ratio) is the probability that the true Sharpe
ratio exceeds zero, given the estimation error implied by sample size, skew,
and kurtosis of daily returns. **DSR** (Deflated Sharpe Ratio, Bailey &
López de Prado 2014) raises the bar: it is the PSR measured against the
Sharpe you would expect from the *best of N random strategies*, where N is
the number of geometry/threshold combinations the machine actually searched
(reported as `n_trials_deflation`). The reporting convention here:
**DSR < 0.95 means the edge is not proven** — the dashboard badges it
plainly rather than hiding it. A machine that can tell you "no edge here" is
the only kind whose "edge here" means anything. An impressive win rate with
a low DSR is a description of the search, not of the market.

## config.yaml tour

| Section     | What it controls                                                        |
| ----------- | ----------------------------------------------------------------------- |
| `universe`  | the 8 symbols scanned                                                   |
| `data`      | cache/DB paths, detection (5min) and refinement (1min) timeframes       |
| `session`   | RTH boundaries: open/close, flat-by, no-new-entries, earliest trigger   |
| `setups`    | trigger definitions: extension vs ATR, wick fraction, cooldown, per-day cap |
| `entry`     | 1-min break-confirmation window and tick size                           |
| `geometry`  | MAE/MFE quantiles searched, vol-regime count, min samples per cell, scan grid |
| `model`     | LightGBM params, walk-forward (min train months, embargo days), threshold grid |
| `learning`  | loss-attribution tree depth, leaf support, loss-rate trigger, threshold bump |
| `execution` | slippage bps, commission/share, risk % per trade, max concurrent, starting equity |
| `export`    | dashboard JSON output directory                                          |

## KPI glossary

| KPI | Exact meaning |
| --- | --- |
| `trades` | count of taken trades that passed the concurrency/sizing filters |
| `win_rate` | % of taken trades with positive net USD P&L |
| `avg_win_usd` / `avg_loss_usd` | mean net USD P&L over winners / losers |
| `avg_win_r` / `avg_loss_r` | mean R-multiple over winners / losers (1R = planned stop distance) |
| `expectancy_r` | mean R-multiple per taken trade — the edge per trade |
| `profit_factor` | gross winnings ÷ gross losses (net of costs) |
| `net_pnl_usd` | total net P&L after slippage and commission, both sides |
| `max_dd_pct` | worst peak-to-trough drawdown of the daily equity curve |
| `sharpe` | annualized Sharpe of daily portfolio returns (√252) |
| `psr` | probability the true Sharpe > 0 given estimation error |
| `deflated_sharpe` | PSR against the best-of-N-trials benchmark; < 0.95 = edge not proven |
| `n_trials_deflation` | geometry × threshold combinations actually searched |

## Repository layout

```
config.yaml            master configuration — single source of truth
src/engine/            the engine (data, setups, labeling, geometry, model,
                       attribution, backtest, metrics, learn, export, cli)
tests/                 synthetic end-to-end smoke test + math unit tests
dashboard/             SvelteKit 2 / Svelte 5 static research terminal
dashboard/static/data/ engine export target (sample data committed)
```
