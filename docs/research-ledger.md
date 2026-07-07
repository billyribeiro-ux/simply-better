# Research ledger

Cumulative record of every search burned against the 2025-01 → 2026-07 range,
what was adopted, what was closed, and what is parked under pre-registration.
The DSR deflation charges `model.external_trials` (config.yaml) on top of the
per-fold geometry × threshold search; keep that number in sync with this file.

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
