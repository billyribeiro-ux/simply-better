"""Period trade reports: the full blotter for any [from, to] window.

The signals come from the honest generator — the full-history walk-forward —
then get filtered to the requested window. A window can only contain signals
for months after the walk-forward's minimum training period; the report says
so plainly when a requested range predates coverage.

Every row carries the explicit trade actions: LONG = BUY at entry, SELL at
exit; SHORT = SELL SHORT at entry, COVER at exit.
"""
from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path

from .config import Config

log = logging.getLogger("engine.report")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _dow(iso: str) -> str:
    return DAYS[date.fromisoformat(iso).weekday()]


def _hm(ts: str) -> str:
    return ts[11:16]


def build_report(artifacts: dict, d_from: date, d_to: date) -> dict:
    """Filter a run's decided signals to the window and compute its stats."""
    lo, hi = d_from.isoformat(), d_to.isoformat()
    window = [s for s in artifacts["signals"] if lo <= s["date"] <= hi]
    taken = [s for s in window if s["taken"]]
    filled = [s for s in taken if s["pnl_usd"] is not None]
    wins = [s for s in filled if s["pnl_usd"] > 0]
    losses = [s for s in filled if s["pnl_usd"] <= 0]

    kpis: dict = {"signals": len(window), "taken": len(taken),
                  "filled": len(filled), "win_rate": None,
                  "net_pnl_usd": 0.0, "expectancy_r": None,
                  "avg_win_usd": None, "avg_loss_usd": None}
    if filled:
        kpis["win_rate"] = round(100.0 * len(wins) / len(filled), 2)
        kpis["net_pnl_usd"] = round(sum(s["pnl_usd"] for s in filled), 2)
        kpis["expectancy_r"] = round(
            sum(s["pnl_r"] for s in filled) / len(filled), 4)
        if wins:
            kpis["avg_win_usd"] = round(sum(s["pnl_usd"] for s in wins) / len(wins), 2)
        if losses:
            kpis["avg_loss_usd"] = round(sum(s["pnl_usd"] for s in losses) / len(losses), 2)

    by_symbol: dict = {}
    for s in filled:
        d = by_symbol.setdefault(s["symbol"], {"trades": 0, "wins": 0, "net": 0.0})
        d["trades"] += 1
        d["wins"] += 1 if s["pnl_usd"] > 0 else 0
        d["net"] = round(d["net"] + s["pnl_usd"], 2)

    rows = []
    for s in window:
        long = s["side"] == "LONG"
        rows.append({
            "date": s["date"], "day": _dow(s["date"]), "symbol": s["symbol"],
            "setup": s["setup"], "side": s["side"],
            "entry_action": "BUY" if long else "SELL SHORT",
            "exit_action": "SELL" if long else "COVER",
            "taken": s["taken"],
            "prob": s["prob"], "threshold": s["threshold"],
            "trigger_time": _hm(s["trigger_ts"]),
            "entry_time": _hm(s["entry_ts"]), "entry_px": s["entry_px"],
            "stop_px": s["stop_px"], "target_px": s["target_px"],
            "exit_time": _hm(s["exit_ts"]), "exit_px": s["exit_px"],
            "exit_reason": s["exit_reason"], "outcome": s["outcome"],
            "pnl_r": s["pnl_r"], "pnl_usd": s["pnl_usd"], "shares": s["shares"],
            "mae_r": s["mae_r"], "mfe_r": s["mfe_r"],
        })

    note = None
    covered = sorted({s["date"] for s in artifacts["signals"]})
    if covered and lo < covered[0]:
        note = (f"walk-forward coverage begins {covered[0]} (the first months "
                "of history are training-only); earlier dates cannot have signals")

    return {"run_id": artifacts["run_id"], "from": lo, "to": hi,
            "kpis": kpis, "by_symbol": by_symbol, "rows": rows, "note": note}


def write_report(rep: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"backtest-{rep['from']}-to-{rep['to']}"
    md_path = out_dir / f"{stem}.md"
    csv_path = out_dir / f"{stem}.csv"

    k = rep["kpis"]
    lines = [
        f"# Backtest blotter — {rep['from']} to {rep['to']}",
        "",
        f"Run `{rep['run_id']}` (EXPLORATORY) · walk-forward out-of-sample: every "
        "decision came from models trained only on data before that trade's month.",
        "",
    ]
    if rep["note"]:
        lines += [f"**Note:** {rep['note']}", ""]
    lines += [
        "| metric | value |", "| --- | --- |",
        f"| signals / taken / filled | {k['signals']} / {k['taken']} / {k['filled']} |",
    ]
    if k["filled"]:
        lines += [
            f"| win rate | {k['win_rate']}% |",
            f"| net P&L | ${k['net_pnl_usd']:,.2f} |",
            f"| expectancy | {k['expectancy_r']:+.3f}R |",
            f"| avg win / avg loss | ${k['avg_win_usd'] or 0:,.2f} / ${k['avg_loss_usd'] or 0:,.2f} |",
        ]
    if rep["by_symbol"]:
        lines += ["", "| ticker | trades | win rate | net P&L |", "| --- | --- | --- | --- |"]
        for sym in sorted(rep["by_symbol"], key=lambda x: -rep["by_symbol"][x]["net"]):
            d = rep["by_symbol"][sym]
            lines.append(f"| {sym} | {d['trades']} | "
                         f"{100.0 * d['wins'] / d['trades']:.0f}% | ${d['net']:,.2f} |")

    lines += ["", "## Trades", "",
              "| date | day | ticker | action | entry | stop | target | exit | via | outcome | R | P&L $ | shares |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in (r for r in rep["rows"] if r["taken"]):
        pnl = f"{r['pnl_usd']:+,.2f}" if r["pnl_usd"] is not None else "— (cap)"
        lines.append(
            f"| {r['date']} | {r['day']} | {r['symbol']} "
            f"| {r['entry_action']} → {r['exit_action']} "
            f"| {r['entry_time']} @ {r['entry_px']:.2f} | {r['stop_px']:.2f} "
            f"| {r['target_px']:.2f} | {r['exit_time']} @ {r['exit_px']:.2f} "
            f"| {r['exit_reason']} | {r['outcome']} | {r['pnl_r']:+.2f} "
            f"| {pnl} | {r['shares'] if r['shares'] is not None else '—'} |")

    skipped = [r for r in rep["rows"] if not r["taken"]]
    lines += ["", f"## Skipped signals ({len(skipped)})", "",
              "Triggers the machine declined (probability below threshold, no "
              "runway, or trend veto). Would-have outcomes shown for research.",
              "",
              "| date | day | ticker | setup | side | trigger | prob | thr | would-have | R |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in skipped:
        lines.append(
            f"| {r['date']} | {r['day']} | {r['symbol']} | {r['setup']} "
            f"| {r['side']} | {r['trigger_time']} | {r['prob']:.3f} "
            f"| {r['threshold']:.3f} | {r['outcome']} | {r['pnl_r']:+.2f} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    fields = list(rep["rows"][0].keys()) if rep["rows"] else [
        "date", "day", "symbol", "setup", "side", "entry_action", "exit_action",
        "taken", "prob", "threshold", "trigger_time", "entry_time", "entry_px",
        "stop_px", "target_px", "exit_time", "exit_px", "exit_reason",
        "outcome", "pnl_r", "pnl_usd", "shares", "mae_r", "mfe_r"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rep["rows"])

    log.info("report -> %s / %s", md_path, csv_path)
    return md_path, csv_path
