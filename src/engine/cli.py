"""CLI: mie ingest | mie run | mie all | mie train | mie live"""
from __future__ import annotations

import logging
import time as time_mod
from datetime import date, datetime, time

import typer

from . import export as export_mod
from . import live as live_mod
from . import paper as paper_mod
from . import production
from . import report as report_mod
from .config import load_config
from .data import ingest as ingest_data
from .learn import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="MIE HOD/LOD self-learning engine")


def _dates(d_from: str, d_to: str) -> tuple[date, date]:
    a, b = date.fromisoformat(d_from), date.fromisoformat(d_to)
    if a >= b:
        raise typer.BadParameter("--from must be before --to")
    return a, b


@app.command()
def ingest(
    date_from: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Pull daily + 5-min + 1-min bars for the universe into the cache."""
    cfg = load_config(config)
    a, b = _dates(date_from, date_to)
    ingest_data(cfg, a, b)
    typer.echo("ingest complete")


@app.command()
def run(
    date_from: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    config: str = typer.Option("config.yaml", "--config"),
    no_export: bool = typer.Option(False, "--no-export"),
) -> None:
    """Full research loop on cached data + dashboard export."""
    cfg = load_config(config)
    a, b = _dates(date_from, date_to)
    artifacts = run_pipeline(cfg, a, b)
    k = artifacts["summary"]
    typer.echo(
        f"\nrun {artifacts['run_id']}  {a} -> {b}\n"
        f"trades={k.get('trades', 0)}  win_rate={k.get('win_rate', 0)}%  "
        f"expectancy={k.get('expectancy_r', 0)}R  "
        f"net={k.get('net_pnl_usd', 0)} USD\n"
        f"sharpe={k.get('sharpe', 0)}  PSR={k.get('psr', 0)}  "
        f"DSR={k.get('deflated_sharpe', 0)}\n"
        f"AUC(test) mean={artifacts.get('diagnostics', {}).get('auc_mean_test')}"
    )
    if not no_export:
        export_mod.write(cfg, artifacts)


@app.command()
def train(
    date_from: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Train the production live-signal model on cached data and save it."""
    cfg = load_config(config)
    a, b = _dates(date_from, date_to)
    bundle = production.train_production(cfg, a, b)
    path = production.save_bundle(cfg, bundle)
    typer.echo(
        f"\nproduction model trained {a} -> {b}\n"
        f"events={bundle.n_events}  labeled={bundle.n_labeled}  "
        f"threshold={bundle.fm.threshold:.3f}  rules={len(bundle.rules)}\n"
        f"geometry cells: "
        + ", ".join(f"{r['setup']} r{r['regime']}={r['anchor']}"
                    for r in bundle.geometry.rows())
        + f"\nsaved -> {path}"
    )


@app.command()
def live(
    date_str: str = typer.Option(None, "--date",
                                 help="replay a past session (YYYY-MM-DD); "
                                      "default = today in US/Eastern"),
    poll: int = typer.Option(0, "--poll",
                             help="re-scan every N seconds until 15:55 ET"),
    no_refresh: bool = typer.Option(False, "--no-refresh",
                                    help="scan the cache without hitting FMP"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Generate live signals for one session with the production model."""
    cfg = load_config(config)
    bundle = production.load_bundle(cfg)
    replay = date_str is not None
    session = date.fromisoformat(date_str) if replay else live_mod.now_et_naive().date()

    while True:
        now_et = (datetime.combine(session, time(16, 0)) if replay
                  else live_mod.now_et_naive())
        if not no_refresh:
            live_mod.refresh_data(cfg, session)
        report = live_mod.scan_live(cfg, bundle, session, now_et)
        live_mod.write_live_json(cfg, report)
        live_mod.persist_live(cfg, report)

        typer.echo(f"\n{session} as of {now_et:%H:%M} ET  "
                   f"(model through {report['trained_through']}, "
                   f"base thr {report['threshold_base']})")
        if report.get("note"):
            typer.echo(report["note"])
        for s in report["signals"]:
            typer.echo(
                f"  {s['symbol']:<5} {s['setup']:<11} {s['side']:<5} "
                f"trig {s['trigger_ts'][11:16]}  p={s['prob']:.3f}/{s['threshold']:.3f} "
                f"{'TAKE ' if s['taken'] else 'skip '}"
                f"entry@{s['entry_trigger_px']:<8} stop {s['stop_px']:<8} "
                f"tgt {s['target_px']:<8} x{s['shares']:<5} [{s['status']}]"
            )

        if replay or poll <= 0:
            break
        if live_mod.now_et_naive().time() >= time(15, 55):
            typer.echo("session flat-by reached — resolving paper trades")
            _resolve_paper(cfg, bundle, session,
                           datetime.combine(session, time(16, 0)))
            break
        time_mod.sleep(poll)


def _resolve_paper(cfg, bundle, session: date, now_et: datetime) -> None:
    rows = paper_mod.resolve_day(cfg, bundle, session, now_et)
    paper_mod.persist_day(cfg, session, rows)
    path = paper_mod.export_record(cfg)
    for r in rows:
        typer.echo(
            f"  PAPER {r['symbol']:<5} {r['side']:<5} "
            f"{r['entry_ts'][11:16]}@{r['entry_px']:<8.2f} -> "
            f"{r['exit_ts'][11:16]}@{r['exit_px']:<8.2f} [{r['exit_reason']:<6}] "
            f"{r['pnl_r']:+.2f}R  ${r['pnl_usd']:+,.2f}"
        )
    if not rows:
        typer.echo("  no taken+confirmed signals to resolve")
    typer.echo(f"  paper record -> {path}")


@app.command()
def paper(
    date_str: str = typer.Option(None, "--date",
                                 help="session to resolve (YYYY-MM-DD); "
                                      "default = today in US/Eastern"),
    no_refresh: bool = typer.Option(False, "--no-refresh",
                                    help="resolve from cache without hitting FMP"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Resolve a session's taken signals into the forward paper record."""
    cfg = load_config(config)
    bundle = production.load_bundle(cfg)
    session = (date.fromisoformat(date_str) if date_str
               else live_mod.now_et_naive().date())
    now_et = (live_mod.now_et_naive() if session == live_mod.now_et_naive().date()
              else datetime.combine(session, time(16, 0)))
    if not no_refresh:
        live_mod.refresh_data(cfg, session)
    typer.echo(f"resolving paper trades for {session} as of {now_et:%H:%M} ET")
    _resolve_paper(cfg, bundle, session, now_et)


@app.command()
def report(
    date_from: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    config: str = typer.Option("config.yaml", "--config"),
    out_dir: str = typer.Option("reports", "--out-dir"),
) -> None:
    """Backtest blotter for a period: every trade with buy/sell action,
    long/short, entry, stop, target, exit, date and time, R and dollars.

    Signals are generated by the full-history walk-forward (the only honest
    generator) and then filtered to the window."""
    from .data import Store
    import polars as pl
    cfg = load_config(config)
    a, b = _dates(date_from, date_to)
    # pipeline history starts at the cache's beginning, ends at the window end
    store = Store(cfg)
    b5 = store.load_bars(cfg.raw["data"]["detect_tf"], cfg.universe[0])
    if b5.height == 0:
        raise typer.BadParameter("no cached data — run: mie ingest first")
    cache_start = b5["ts"].min().date()
    typer.echo(f"walk-forward over {cache_start} -> {b} (window {a} -> {b})")
    artifacts = run_pipeline(cfg, cache_start, b)
    rep = report_mod.build_report(artifacts, a, b)
    md, csv_ = report_mod.write_report(rep, cfg.root / out_dir)
    k = rep["kpis"]
    if rep["note"]:
        typer.echo(f"note: {rep['note']}")
    typer.echo(
        f"window {a} -> {b}: signals={k['signals']} taken={k['taken']} "
        f"filled={k['filled']}"
        + (f"  win_rate={k['win_rate']}%  net=${k['net_pnl_usd']:,.2f}  "
           f"expectancy={k['expectancy_r']:+.3f}R" if k["filled"] else "")
    )
    typer.echo(f"blotter -> {md}\ncsv     -> {csv_}")


@app.command()
def daily(
    config: str = typer.Option("config.yaml", "--config"),
    train_from: str = typer.Option("2025-01-02", "--train-from",
                                   help="history start for the nightly retrain"),
) -> None:
    """Hands-free daily loop (run after the close):

    1. refresh market data (bootstraps the full history on a cold cache);
    2. resolve today's paper trades with the bundle that actually generated
       them (never the retrained one — record integrity);
    3. retrain the production model through today for tomorrow's session;
    4. publish the live + paper artifacts.
    """
    cfg = load_config(config)
    today = live_mod.now_et_naive().date()
    start = date.fromisoformat(train_from)
    typer.echo(f"daily loop for {today}")

    # 1. data: cold caches (fresh CI runners) get the full history
    from .data import Store, ingest as ingest_data
    store = Store(cfg)
    b5 = store.load_bars(cfg.raw["data"]["detect_tf"], cfg.universe[0])
    n_days = (b5.select(b5["ts"].dt.date().alias("d")).unique().height
              if b5.height else 0)
    if n_days < 140:
        typer.echo(f"cache has {n_days} sessions — full bootstrap ingest")
        ingest_data(cfg, start, today)
    else:
        live_mod.refresh_data(cfg, today)

    # 2. resolve today's paper record with the bundle that traded it
    now_et = live_mod.now_et_naive()
    try:
        bundle = production.load_bundle(cfg)
        report = live_mod.scan_live(cfg, bundle, today, now_et)
        live_mod.write_live_json(cfg, report)
        live_mod.persist_live(cfg, report)
        _resolve_paper(cfg, bundle, today, now_et)
    except RuntimeError as exc:
        typer.echo(f"no usable bundle for paper resolution ({exc}) — "
                   "first run trains one below")

    # 3. retrain through today -> tomorrow's bundle
    new_bundle = production.train_production(cfg, start, today)
    production.save_bundle(cfg, new_bundle)
    typer.echo(f"retrained through {today}: labeled={new_bundle.n_labeled} "
               f"thr={new_bundle.fm.threshold:.3f} rules={len(new_bundle.rules)}")
    typer.echo("daily loop complete")


@app.command("all")
def run_all(
    date_from: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    config: str = typer.Option("config.yaml", "--config"),
) -> None:
    """ingest + run + export in one shot."""
    cfg = load_config(config)
    a, b = _dates(date_from, date_to)
    ingest_data(cfg, a, b)
    artifacts = run_pipeline(cfg, a, b)
    export_mod.write(cfg, artifacts)
    typer.echo("done")


if __name__ == "__main__":
    app()
