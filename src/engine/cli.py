"""CLI: mie ingest | mie run | mie all | mie train | mie live"""
from __future__ import annotations

import logging
import time as time_mod
from datetime import date, datetime, time

import typer

from . import export as export_mod
from . import live as live_mod
from . import production
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
            typer.echo("session flat-by reached — stopping")
            break
        time_mod.sleep(poll)


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
