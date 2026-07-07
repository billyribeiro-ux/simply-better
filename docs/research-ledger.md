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
