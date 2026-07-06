"""CLI: mie ingest | mie run | mie all"""
from __future__ import annotations

import logging
from datetime import date

import typer

from . import export as export_mod
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
        f"DSR={k.get('deflated_sharpe', 0)}"
    )
    if not no_export:
        export_mod.write(cfg, artifacts)


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
