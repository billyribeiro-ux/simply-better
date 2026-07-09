# Research ledger

Cumulative record of every search burned against the 2025-01 → 2026-07 range,
what was adopted, what was closed, and what is parked under pre-registration.
The DSR deflation charges `model.external_trials` (config.yaml) on top of the
per-fold geometry × threshold search; keep that number in sync with this file.

## 2026-07-08 — ROOT-CAUSE AUDIT: the edge is real but thin and execution-bound

Owner ordered a deep end-to-end audit. 8 independent lenses (data, labels,
features, model, edge-existence, geometry, cost, simulation), each adversarially
verified; I reproduced the decisive numbers myself. Verdict, with hard evidence:

**The pipeline is not broken and the edge is not absent.** Confirmed clean:
labels reproduce BY HAND for all 1013 trades with zero mismatches; data faithful
(NVDA split consistent, no bad ticks); 23 features hand-recompute exactly, no
NaN/constant/Int8 bug; LightGBM trains correctly (AUC~0.51 is the honest
signature of a near-random-GIVEN-features target, not a training defect); the
event-driven accounting fix reconciles to the cent with zero residual lookahead.

**The real root cause is a THIN edge sitting on the transaction-cost boundary.**
Run 40ff77b9cc42 (concept-2 breaks), net vs slippage (self-verified by
recomputation from raw fills, commission 0.0035/sh both sides):

| slippage (bps/side) | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
| ------------------- | - | --- | --- | --- | --- |
| net USD | +35,832 | +25,229 | +14,627 | +4,024 | **−6,578** |

**Break-even slippage = 1.69 bps/side (concept-2); 1.38 (v3).** The backtest's
flat 2.0 bps/side assumption is just past break-even — that single arbitrary
default, not any absence of signal, is what turns the gross edge negative.

Is 2.0 bps/side realistic? Corwin-Schultz (2012) high-low effective spread on
the 5-min bars (an UPPER bound — 5-min H/L includes intrabar vol, so true quoted
spreads are tighter): SPY 0.7, QQQ 0.9, IWM 0.9, AAPL 1.3, AMZN 1.5, NFLX 1.4,
NVDA 2.3, TSLA 2.9 bps/side. So the flat 2.0 is conservative-to-punitive for the
five liquid names and roughly right only for NVDA/TSLA. A desk with competent
execution (limit/mid, liquidity provision, smart routing) trades the liquid
names well under break-even; a retail market order at 2-3 bps does not. **The
strategy's viability is literally an institutional-execution question.**

Honest caveats recorded WITH the finding (do not oversell):
- DSR ~0 even at ZERO cost — the gross edge is on heavily-mined 2024-2026
  development data; the deflator does not certify it. Needs true OOS + live
  execution measurement before any dollar claim.
- Per-trade edge is tiny (~0.02 ATR, ~$14-35/trade at low cost) — thin and
  fragile; a knife-edge on cost is not a robust business.
- Corwin-Schultz is a proxy; real all-in cost adds impact (small at ~500-sh
  size on these names) and short borrow (negligible intraday). Verify live.
Directional consequences: lowering the cost assumption to make P&L positive
would be "a prettier number by loosening" and is BANNED as tuning. The
legitimate moves are (1) measure true per-name execution cost and gate trades on
net-of-REALISTIC-cost EV, and (2) pursue larger, less cost-fragile edges
(cross-sectional / statistical-relationship strategies) — pre-registered.

## 2026-07-08 — CORRECTION: entry-vs-exit accounting leak overturns the USD record

A parallel-agent audit (cycle 1 of the self-paced research loop) found a
lookahead bug in `backtest.run` and I verified it independently before acting.
The simulator credited each trade's full realized PnL to `equity` and to the
day's running loss at ENTRY-processing time, while the position's exit is hours
later. Since 74.4% of trades enter while an earlier position is still open, the
equity that SIZED each trade and the day-loss that ARMED the kill switch already
contained the FUTURE outcomes of open positions. Same invariant class as the
iteration-one slippage-sign inversion. Fixed: event-driven accounting, PnL
realized only at `exit_ts` (commit "Fix lookahead in portfolio sim"), regression
tests pin the invariant.

**What it does to the record.** R-space expectancy is share-independent and
survives — but the taken SET changes, because the old kill switch was silently
deleting future-known losers. Honest re-runs over the identical frozen spec,
2024-01-02 → 2026-07-07, embargoed walk-forward:

| spec (honest, exit-time accounting) | trades | expectancy_r | 95% CI | net USD | Sharpe |
| ----------------------------------- | ------ | ------------ | ------ | ------- | ------ |
| concept-2 (breaks only)  run 40ff77b9cc42 | 1013 | +0.0862R | [+0.019, +0.155] | **−$6,578**  | −0.168 |
| v3 (breaks + ORB)        run 016e5fae102e | 2146 | +0.0797R | [+0.031, +0.128] | **−$27,523** | −0.545 |
| v3, kill switch OFF      run c9ea733d75dc | 2393 | +0.0733R | —                | **−$35,677** | −0.722 |

Decomposition of v3 old→new (28cb1337c4d8 → 016e5fae102e): 165 trades the buggy
kill switch used to suppress are re-admitted at **−0.271R / −$26,721** — the days
the sim previously "knew" were bad. That single set is the whole swing.

**Honest verdicts, applied per the letter of each pre-registration:**
- **concept-3 ORB → outcome (c): REJECTED.** Joint net (−$27,523) is required to
  be > 0 and must not degrade below concept-2's record; it does both. ORB
  degrades concept-2's −$6,578 to −$27,523 (ORB_UP alone −$32,148). Both ORB
  setups disabled in config. The v3 "adoption" of 2026-07-08 is VACATED — it
  rested on the contaminated +$7,212 number.
- **concept-2 breaks → outcome (b): WEAK, research bench.** Its pre-registration
  triggers (b) on `net ≤ 0`. Honest net is −$6,578. The R-edge is real
  (+0.086R, CI excludes zero) but does NOT survive transaction costs under the
  current tight-stop geometry. Live firing disabled; the concept stays in code
  for the cost-structure experiment below.
- **The engine currently ships NO net-profitable live edge.** Every prior
  positive USD record (v1 +$1,154, v2/v3 +$7,212) was an accounting artifact.
  Stated plainly and without hedging: after honest costs, this strategy as
  specified loses money. The signal has genuine pre-cost R-predictive value;
  converting it to positive dollars is unsolved.

Config change: `hod_break`, `lod_break`, `orb_up`, `orb_down` all `enabled:
false`. `mie live`/`mie daily` therefore emit no signals until a spec clears a
pre-registered net-of-cost criterion. The production bundle on disk (v3) is
inert (detection disabled) and superseded.

## PRE-REGISTERED: cost-structure experiment (declared 2026-07-08, before any run)

Diagnosis fixed a priori: the real +0.086R edge dies because 0.2-ATR stops
produce ~500-share positions whose 2 bps slippage + commission (~$40-45/trade,
≈0.06R) exceeds the edge. The geometry search optimizes R (frictionless), so it
is blind to this. Experiment (parameters fixed by theory, not fitted): change
the geometry objective to **net-of-cost expectancy** — subtract the modeled
round-trip cost (slippage_bps + commission, from config execution) converted to
R at each candidate stop width, then re-run the frozen walk-forward. No other
change; no per-cell hand-tuning. Pre-registered read of the single run:
- (a) net-of-cost geometry yields concept-2 joint net > 0 AND expectancy_r still
  > +0.05 → adopt as the shipping spec (v4), forward-EXPLORATORY;
- (b) net > improves but stays ≤ 0 → the edge is real but sub-cost on this
  universe; document and keep on the bench (no live spec);
- (c) expectancy_r ≤ +0.05 after the change → the tight-stop R-edge was itself
  fragile; concept-2 closed.
Design DOF charged: +2 external trials (178 → 180). This is the next loop cycle.

**VERDICT (cycle 2, 2026-07-08): the cost-structure fix does NOT rescue the
edge — outcome (b), effectively (c). No live spec.** `geometry.net_of_cost`
implemented (cost in R = 2·(slip·px+comm)/(stop·atr), share count cancels;
unit-tested to the 6th decimal) and the frozen walk-forward re-run:

| spec (net-of-cost geometry) | trades | expectancy_r | 95% CI | net USD |
| --------------------------- | ------ | ------------ | ------ | ------- |
| concept-2 (breaks)  run 183e407e7628 | 638  | +0.0511R | **[−0.026, +0.133]** | −$7,296  |
| v3 (breaks + ORB)   run 4fab7c16da69 | 1059 | +0.0623R | —                    | −$12,482 |

The geometry did exactly what it was built to do — it abandoned the uniform
0.2-ATR stop for a 0.3–0.7-ATR spread to cut the friction drag. But the
decisive fact is the confidence interval: concept-2's edge went from +0.086R
(CI **excludes** zero) under frictionless geometry to +0.051R with a CI that
**straddles** zero. Widening the stop to make each trade affordable dilutes the
per-trade edge until it is statistically indistinguishable from noise, and net
stays negative (−$7,296, no improvement on the −$6,578 frictionless-geometry
concept-2). Criterion (a) fails on `net ≤ 0`; the CI-includes-zero result is the
(c)-flavoured reality: the tight-stop R-edge was fragile.

Honest bottom line after two loop cycles: **on this 8-symbol universe, the
HOD/LOD break + ORB families do not constitute a net-profitable intraday
strategy after honest transaction costs.** The prior positive records were an
accounting artifact (cycle 1); the residual real edge does not survive its own
costs (cycle 2). `net_of_cost` stays in the codebase (config, default false) as
a validated capability; no spec adopts it. Setups remain disabled.

## PRE-REGISTERED: cycle 3 — model discrimination + net-EV trade selection (declared 2026-07-08)

Rationale: the meta-model AUC is ~0.511 — it barely separates winners from
losers, so the decision layer takes essentially the unconditional event stream,
whose edge is sub-cost (cycles 1-2). The unexplored lever is DISCRIMINATION: if
the model could identify the SUBSET of triggers whose expected net-of-cost R is
positive, the strategy could trade only those and be profitable even though the
unconditional edge is not. Two a-priori changes, no tuning on this range:
(1) add the point-in-time-safe features the fleet's feature-eng lens proposes
(re-run that lens under the current model — it died on the session limit);
(2) replace the fixed probability threshold with a NET-EV GATE: take a trade
only when `p·(t/s) − (1−p) − cost_R > 0` using the calibrated p and the cell's
own geometry, so selection is denominated in expected dollars-after-cost, not a
bare probability. Pre-registered read of the single walk-forward run:
- (a) net-EV gate yields concept-2 joint net > 0 with expectancy_r CI excluding
  zero → adopt (v4), forward-EXPLORATORY;
- (b) net improves toward 0 but stays ≤ 0, or CI still includes zero → the
  discrimination is insufficient; document and keep benched;
- (c) no improvement over cycle-2 net → the model carries no exploitable
  conditional signal on this universe; escalate to a universe/timeframe review.
DOF charged at the run (feature count + gate = fixed a priori); update
external_trials then.

## Trials charged (2026-07 multi-agent research program)

| program            | variants | outcome |
| ------------------ | -------- | ------- |
| entry-timing       | 11       | lever CLOSED — touch confirmation is the causal optimum; every alternative −0.002R to −0.099R/order. Close-confirmation worse in 19/19 months: banned. |
| loss-filters       | 20       | `session.no_new_after: "14:50"` adopted as window hygiene at ZERO expectancy credit (dose-response replicated in runs 379501eeebdd / 013398a8b857). All other filters rejected: wick/VWAP-band/early-session cuts remove positive-expectancy events; room floors are anti-signal. |
| features-v2        | 19       | deferred one cycle: `trig_range_atr` (+0.015 pooled OOS AUC, 8/12 months) queued for a clean gate-on ablation next iteration; screening labels had a disclosed production-geometry leak, deflated p ~0.27-0.5. All other candidates rejected for the model. |
| per-symbol gating  | 38       | parked indefinitely: symbol skill not persistent (trailing-vs-next Spearman +0.027, p=0.82). Pre-registered frozen config: X=−0.20, window=6mo, min_trades=8; evaluate only on post-2026-07 forward months and only if rank persistence first appears (Spearman >+0.2, p<0.05). Per-setup thresholds: dead, do not implement on this record. |
| exit-engineering   | 20       | no exit rule adopted (time-stops forfeit the EOD bucket — the best bucket at +0.19-0.22R on 41% of trades; breakeven moves negative at best-case bound). PathScan path instrumentation landed for the next experiment: underwater time-stop (bounded [−0.024,+0.058]R) and target-side geometry. |
| regime-gating      | 40       | ADOPTED: day-type trend veto `setups.day_type.eff_gate_max: 0.45` (frozen). Drops trades whose aligned first-hour efficiency fades a trending session: drop bucket −0.192R/trade (CI95 [−0.39,−0.03], P(<0)=0.992), 9/13 months improved, orthogonal to model prob (corr −0.115) and to gk_vol_z. Kept bucket ≈ breakeven — this is a protective veto, not proven edge. |
| **total charged**  | **148**  | `model.external_trials: 148` |

## Adopted (2026-07 slate)

1. Day-type trend veto (decision layer only — never a detection filter, so the
   event stream/cooldowns are unchanged and geometry/model still train on all
   events). Applied identically in `learn.run_research` and `live.scan_live`.
2. `no_new_after` 15:00 → 14:50 (gates on trigger-bar OPEN timestamp).
3. Slippage sign fix in `backtest.py` — costs had been credited in the
   trade's favor since iteration one (invariant 7 violation). All prior
   USD/Sharpe/DSR records are optimistic by roughly 4 bps of notional per
   round trip; R-based expectancy was unaffected (frictionless).
4. PathScan `fav_path/adv_path/close_path` instrumentation (research only).

## Pre-registered / parked (forward data only — no retuning on this range)

- vwap-stretch filter `side*dist_vwap_atr < −1.0` (frozen at 1.0 ATR).
- symbol gate (config above).
- `trig_range_atr` feature ablation (gate-on baseline vs gate-on+feature).
- Gate sub-path audit: the pre-10:30 efficiency-so-far fallback carries most
  of the veto's effect (−0.357R × 61) vs frozen-h1 (−0.107R × 118); monitor
  separately on forward months.
- Re-audit ALL of the above after ≥3 months of frozen-spec forward walk-forward.

## Second experiment cycle (2026-07-07, post-slate)

| experiment | variants | outcome |
| --- | --- | --- |
| selection-layer audit | 2 | **closed, positively**: on the gate-on OOS record, taken trades run +0.051R vs −0.012R all-decided and −0.131R low-prob skips; per-fold IC(prob, pnl) +0.152, 9/13 folds positive; "no threshold" counterfactual is WORSE (−0.028R). The model+threshold layer adds ~+0.08R/trade of selection value. The earlier "anti-selection" alarm was a geometry-vintage artifact. |
| underwater time-stop | 9 | **dead, exactly**: with close_path prices, every (T, gap) ∈ {45,60,90}×{0,0.2,0.4} is negative (−0.001..−0.044R, ≤7/19 months positive). The research program's bounded midpoint (+0.015R) did not survive exact evaluation. |
| closer anchor fracs (0.25, 0.382 added to the per-fold search) | 8 | **rejected by pre-registered criterion**: expectancy +0.021 vs baseline +0.080, net −$10.8k vs −$2.4k. The "targets too far" hypothesis does not convert through the honest harness. Not adopted; not committed. |
| trig_range_atr ablation | 1 | **rejected by pre-registered criterion** (run c68753bbb833): AUC(test) mean 0.5535 vs 0.5763 gate-on baseline — the screened +0.015 gain reversed under fold-honest per-fold geometry, exactly the deflation objection raised in verification. Expectancy flat (+0.0025). Feature code reverted; candidate is dead for the model on this record. |

Trials charged: external_trials 148 → **168** (+2 audit, +9 time-stop, +8 fracs, +1 ablation, +2 rounding reserve).

## PRE-REGISTERED: the 2024 holdout test (declared 2026-07-07, before results)

FMP serves 1-minute data through 2024. No search, tuning, or selection in
this program has ever touched 2024 — every spec value (gate 0.45,
no_new_after 14:50, geometry/threshold grids, external trials) was chosen
on 2025-01→2026-07 only. Running the frozen spec over 2024-01→2026-07
makes the 2024-H2 walk-forward folds (models trained on 2024 data only,
per the expanding window) an untouched holdout.

Pre-registered read, stated before the run:
- (a) 2024-H2 taken-trade expectancy > 0 on ≥ 50 trades → first positive
  untouched-data evidence; strategy stays live, forward record continues.
- (b) expectancy ≤ 0 → third independent replication of no-edge, now on
  virgin data; the setup CONCEPT (not its tuning) goes under review.
Either way the full 30-month history feeds subsequent production training.

Caveat recorded: 2024 is untouched by the *tuning*, but the setup concept
was designed by people who knew markets generally — this is weaker than
true forward data, stronger than anything else available today.

**VERDICT (run 8fe8485638ff, 25 folds over 30 months): outcome (b).**
2024-H2 holdout: 134 trades, expectancy **−0.039R**, win rate 45.5%,
net −$7,801, 4/6 months positive but negative overall. The mined range
itself softened to +0.004R with 2024 in the training windows. Full
30-month record: expectancy 0.0003R [CI −0.085, +0.084], net −$21.8k
after honest costs, max DD −22.2%, DSR 0.0 at 14,843 trials.

**Standing conclusion: the HOD-fade / LOD-reclaim concept, as specified,
has no edge on this universe — replicated three times independently,
now including virgin data.** Per the pre-registration, the setup concept
is under review. The platform, validation harness, live path, and paper
record are sound and ready for the next concept; no further tuning of
THIS concept on ANY historical range is permitted. The forward paper
record may continue for monitoring, with expectations set accordingly.

## PRE-REGISTERED: concept 2 — trend continuation (declared 2026-07-07, before any run)

Derived from concept 1's failure evidence, not from new mining: the trend
veto's drop bucket (fading trending sessions) ran −0.19R; never-confirming
fade events averaged −0.46R (the extreme kept extending); loss clusters sit
in momentum conditions; reversion runway was never traversed. The mirror
concept: HOD_BREAK (long a strong new-high bar when the session already
trends up), LOD_BREAK (short mirror).

Parameters fixed A PRIORI, zero tuning: ext_min_atr 0.35 (mirrored from the
fade spec), body_frac_min 0.5 (close in the break-side half of the bar),
eff_min 0.30 (session efficiency aligned with the break). Fades disabled.
Design degrees of freedom charged: +6 external trials (168 → 174).

Pre-registered read of the single 30-month walk-forward run:
- (a) expectancy_r > +0.05 AND net_pnl_usd > 0 after honest costs →
  promising; concept goes live on the forward paper record (EXPLORATORY —
  2024-25-26 is development data; only forward months confirm).
- (b) 0 < expectancy_r ≤ +0.05 or net ≤ 0 → weak; keep on research bench.
- (c) expectancy_r ≤ 0 → trend continuation as specified is also dead;
  the universe/timeframe itself goes under review.
No parameter may be revisited on this range regardless of outcome.

> **CORRECTED 2026-07-08** — the +$1,154 net below is contaminated by the
> entry-vs-exit accounting leak (top-of-file CORRECTION). Honest re-run
> (40ff77b9cc42): +0.0862R but **net −$6,578**, so concept-2 falls to outcome
> (b) research bench, not the (a) live-EXPLORATORY promotion recorded here. The
> R-edge is real; it does not survive costs. Preserved for audit.

**VERDICT (run 65603170d228, 25 folds / 24 OOS months): outcome (a).**
998 filled trades, expectancy **+0.0997R with 95% CI [+0.027, +0.172]** —
the first zero-excluding positive interval in this program — 17/24 months
positive, net **+$1,154** after honest slippage and commission.
Per-setup: LOD_BREAK +0.130R (462 trades) > HOD_BREAK +0.054R (649) —
downside momentum stronger. Geometry chose ATR mode in all six cells
(reversion anchors correctly lost the search for continuation trades),
converging on tiny 0.2-ATR stops with 0.3–0.5 targets.

Honest counterweights, recorded with the win: profit factor 1.005 — the
tiny-stop geometry produces large share counts, so costs consume nearly
the entire R edge in dollars; Sharpe 0.13 [CI −1.10, +1.35]; DSR 0.0001
at 14,849 trials (the deflator still refuses certification, as it should
after this much mining). AUC 0.514: the meta-model adds little on
breakouts — the raw setup carries the edge.

Per pre-registration: concept 2 goes live on the forward paper record,
EXPLORATORY. Standing questions for FORWARD DATA ONLY (no retuning):
(1) does the R edge persist out of development data; (2) the cost-drag
structure (stop width vs share count) may only be revisited as a
pre-registered experiment after ≥3 months of forward record. The paper
record now spans two concepts; trades are distinguishable by setup.

## PRE-REGISTERED: concept 3 — opening range breakout (declared 2026-07-08, before any run)

Motivation: coverage. The tape prints ~19% of cumulative daily range across
the universe; two concepts capture a sliver of the day's motion. ORB fires
in one or both directions on most sessions for most symbols — the highest
signal-frequency classic pattern. Parameters fixed A PRIORI, zero tuning:
opening range = first 30 minutes (bars closing <= 10:00); ORB_UP triggers
on the first 5-min bar CLOSING above the OR high with close > open
(ORB_DOWN mirrors below the OR low); standard 1-min confirmation,
cooldown, and per-side caps; no additional filters. Trend veto applies at
decision time as usual (it suppresses counter-trend breaks). Design DOF
charged: +4 external trials (174 → 178).

Spec-revision note, recorded honestly: enabling a new setup family changes
the joint model and geometry, so the frozen-spec forward clock RESTARTS at
adoption (v3). The paper record continues uninterrupted — trades are
tagged by setup and trained_through — but forward-confirmation counting
for the joint spec begins anew. This is a deliberate trade-off ordered by
the owner: coverage now, at the price of a longer confirmation runway.

Pre-registered read of the single 30-month walk-forward run (breaks + ORB
jointly, the shipping configuration):
- (a) joint expectancy_r > +0.05 AND joint net > 0 AND the ORB setups'
  own expectancy > 0 → ORB joins the live spec (v3);
- (b) joint holds but ORB's own expectancy ≤ 0 → ORB stays disabled;
  concept-2 spec unchanged (clock not reset);
- (c) joint degrades below concept-2's record → ORB rejected outright.
No ORB parameter may be revisited on this range regardless of outcome.

> **VACATED 2026-07-08** — this verdict rested on run 28cb1337c4d8, whose
> +$7,212 net came from the entry-vs-exit accounting leak (see the CORRECTION
> section at the top of this file). Under honest accounting the joint net is
> −$27,523 and ORB is REJECTED per outcome (c). The text below is preserved as
> the contaminated record it was, not deleted, so the mistake stays auditable.

**VERDICT (run 28cb1337c4d8, 25 folds / 24 OOS months): outcome (a) — v3 adopted.**
Joint record: 2,004 filled trades, expectancy **+0.1131R, 95% CI
[+0.067, +0.161]** (tighter than concept-2's), net **+$7,212** after
honest costs (6× concept-2's +$1,154 on the identical range), Sharpe
0.284 vs 0.133, PSR 0.647 vs 0.564. ORB's own expectancy +0.0986R with
CI [+0.035, +0.157] — zero-excluding, and each direction separately
excludes zero (ORB_UP +0.075R [+0.005, +0.145] n=824; ORB_DOWN +0.140R
[+0.033, +0.248] n=477). All three (a) conditions hold. Coverage: 4.63
trades per traded day across the tape, up from ~2.3.

Honest counterweights, recorded with the win:
- **ORB's own cost-charged dollars are NEGATIVE: −$9,875** over 1,301
  trades, concentrated in ORB_UP (−$17,878; ORB_DOWN +$8,003). The
  R-vs-USD divergence is the known tight-stop friction drag (~0.10R per
  trade at 2 bps + commission on 0.2-ATR stops). The joint dollar
  result still improves because the model/veto layer and LOD_BREAK's
  +$30,671 carry the book. Per the standing concept-2 rule, the cost
  structure (stop width vs share count) may only be revisited as a
  pre-registered experiment after ≥3 months of forward record — no
  retroactive tuning, and no post-hoc cherry-pick of ORB_DOWN alone.
- **Max drawdown deepened to −23.3%** from −12.3% — more trades at
  constant per-trade risk buy coverage at the price of deeper aggregate
  drawdown.
- DSR 0.0001 at 29,253 trials: the deflator still refuses
  certification. This remains an EXPLORATORY edge on development data;
  only the forward record confirms.
Per the spec-revision note above, the frozen-spec forward clock
RESTARTS at v3 (production retrained on the joint spec through
2026-07-07). The paper record continues uninterrupted, tagged by setup.

## Known record caveats

- **82/465 event drift**: 82 trades of run f4e6c58b4603 do not rematch a
  rebuild under the production bundle. Root cause: geometry vintage — each
  fold's geometry (trained on months < fold) marks a slightly different
  tradable set than the full-history production geometry; the cache also
  gained 2026-07-06+ sessions after that run. Consequence: any analysis that
  joins by event to a *different geometry vintage* is favorably selected
  (unmatched trades averaged −0.228R). Event-joined research must rebuild
  labels under the same vintage it evaluates.
- All USD results before the slippage fix (runs ≤ f4e6c58b4603) are
  optimistic; the fix makes subsequent equity curves comparably worse.
- Open question ranked highest by the program synthesis: the selection layer
  itself (calibration + tail-threshold procedure) may be anti-selecting —
  the labeled universe runs ~+0.09R pre-cost while the model-selected taken
  stream ran −0.036R frictionless. Audit before adding any new feature.
