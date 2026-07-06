"""Data layer: FMP ingestion -> Parquet cache -> DuckDB persistence.

All intraday timestamps from FMP are US/Eastern wall-clock. They are kept
naive-ET throughout the engine; the session filter guarantees RTH only.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import polars as pl

from .config import Config

log = logging.getLogger("engine.data")

_BASE = "https://financialmodelingprep.com/stable"

# window sizes per timeframe keep each request comfortably under FMP row caps
# (stable API truncates to the most recent ~500-780 rows per response)
_CHUNK_DAYS = {"1min": 2, "5min": 4}


class FMPClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._key = api_key
        self._client = httpx.Client(timeout=timeout)

    def _get(self, url: str, params: dict[str, str]) -> list | dict:
        params = {**params, "apikey": self._key}
        for attempt in range(5):
            try:
                r = self._client.get(url, params=params)
                if r.status_code == 429:
                    wait = 2.0 * (attempt + 1)
                    log.warning("FMP rate limit; sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as exc:
                if attempt == 4:
                    raise
                wait = 1.5 ** attempt
                log.warning("FMP error %s; retry in %.1fs", exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # ---- intraday ---------------------------------------------------------
    def intraday(self, symbol: str, tf: str, d_from: date, d_to: date) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        step = timedelta(days=_CHUNK_DAYS[tf])
        cur = d_from
        while cur <= d_to:
            end = min(cur + step - timedelta(days=1), d_to)
            rows = self._get(
                f"{_BASE}/historical-chart/{tf}",
                {"symbol": symbol, "from": cur.isoformat(), "to": end.isoformat()},
            )
            if isinstance(rows, list) and rows:
                frames.append(pl.DataFrame(rows))
            cur = end + timedelta(days=1)
        if not frames:
            return _empty_bars()
        df = pl.concat(frames, how="vertical_relaxed")
        df = (
            df.select(
                pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("ts"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            )
            .unique(subset=["ts"], keep="first")
            .sort("ts")
        )
        # RTH only: 09:30 <= ts < 16:00 ET
        # dt.hour()/dt.minute() are Int8 in polars >= 1.30; cast before the
        # *60 arithmetic or it overflows and the filter drops every bar
        mins = (
            pl.col("ts").dt.hour().cast(pl.Int32) * 60
            + pl.col("ts").dt.minute().cast(pl.Int32)
        )
        df = df.filter((mins >= 9 * 60 + 30) & (mins < 16 * 60))
        return df.with_columns(pl.lit(symbol).alias("symbol"))

    # ---- daily ------------------------------------------------------------
    def daily(self, symbol: str, d_from: date, d_to: date) -> pl.DataFrame:
        payload = self._get(
            f"{_BASE}/historical-price-eod/full",
            {"symbol": symbol, "from": d_from.isoformat(), "to": d_to.isoformat()},
        )
        # stable API returns a flat list; legacy wrapped it in {"historical": []}
        hist = payload if isinstance(payload, list) else payload.get("historical", [])
        if not hist:
            return pl.DataFrame(
                schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64,
                        "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
            )
        df = pl.DataFrame(hist)
        return (
            df.select(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            )
            .unique(subset=["date"], keep="first")
            .sort("date")
        )

    def close(self) -> None:
        self._client.close()


def _duck_type(arrow_type: str) -> str:
    """Map an arrow dtype string to a DuckDB column type for ALTER TABLE."""
    t = arrow_type.lower()
    if "timestamp" in t:
        return "TIMESTAMP"
    if t.startswith("date"):
        return "DATE"
    if "int" in t:
        return "BIGINT"
    if "float" in t or "double" in t or "decimal" in t:
        return "DOUBLE"
    if "bool" in t:
        return "BOOLEAN"
    return "VARCHAR"


def _empty_bars() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"ts": pl.Datetime, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
                "symbol": pl.Utf8}
    )


class Store:
    """Parquet cache for bars + DuckDB for run artifacts."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cache = cfg.cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- parquet ----------------------------------------------------------
    def _path(self, tf: str, symbol: str) -> Path:
        d = self.cache / tf
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}.parquet"

    def save_bars(self, tf: str, symbol: str, df: pl.DataFrame) -> None:
        p = self._path(tf, symbol)
        if p.exists():
            old = pl.read_parquet(p)
            df = (
                pl.concat([old, df], how="vertical_relaxed")
                .unique(subset=["ts"], keep="last")
                .sort("ts")
            )
        df.write_parquet(p)

    def load_bars(self, tf: str, symbol: str,
                  d_from: date | None = None, d_to: date | None = None) -> pl.DataFrame:
        p = self._path(tf, symbol)
        if not p.exists():
            return _empty_bars()
        df = pl.read_parquet(p)
        if d_from is not None:
            df = df.filter(pl.col("ts") >= datetime.combine(d_from, datetime.min.time()))
        if d_to is not None:
            df = df.filter(pl.col("ts") < datetime.combine(d_to + timedelta(days=1),
                                                           datetime.min.time()))
        return df.sort("ts")

    def save_daily(self, symbol: str, df: pl.DataFrame) -> None:
        p = self.cache / "daily"
        p.mkdir(parents=True, exist_ok=True)
        fp = p / f"{symbol}.parquet"
        if fp.exists():
            old = pl.read_parquet(fp)
            df = (
                pl.concat([old, df], how="vertical_relaxed")
                .unique(subset=["date"], keep="last")
                .sort("date")
            )
        df.write_parquet(fp)

    def load_daily(self, symbol: str) -> pl.DataFrame:
        fp = self.cache / "daily" / f"{symbol}.parquet"
        if not fp.exists():
            return pl.DataFrame(
                schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64,
                        "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
            )
        return pl.read_parquet(fp).sort("date")

    # ---- duckdb persistence ------------------------------------------------
    def persist_run(self, run_id: str, trades: pl.DataFrame,
                    summary_json: str, geometry_json: str, attribution_json: str) -> None:
        con = duckdb.connect(str(self.cfg.db_path))
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT current_timestamp,
                    summary JSON, geometry JSON, attribution JSON
                )""")
            con.execute(
                "INSERT OR REPLACE INTO runs (run_id, summary, geometry, attribution) "
                "VALUES (?, ?, ?, ?)",
                [run_id, summary_json, geometry_json, attribution_json],
            )
            if trades.height > 0:
                con.register("trades_df", trades.to_arrow())
                con.execute("CREATE TABLE IF NOT EXISTS trades AS SELECT "
                            "CAST(NULL AS VARCHAR) AS run_id, * FROM trades_df LIMIT 0")
                # append-only schema evolution: when a later engine version
                # adds columns, widen the table (old rows keep NULLs) instead
                # of failing the research record
                existing = {r[1] for r in
                            con.execute("PRAGMA table_info('trades')").fetchall()}
                for name, dtype in zip(trades.columns, trades.to_arrow().schema.types):
                    if name not in existing:
                        con.execute(
                            f'ALTER TABLE trades ADD COLUMN "{name}" {_duck_type(str(dtype))}'
                        )
                cols = ", ".join(f'"{c}"' for c in trades.columns)
                con.execute(
                    f"INSERT INTO trades (run_id, {cols}) "
                    f"SELECT '{run_id}', * FROM trades_df"
                )
        finally:
            con.close()


def ingest(cfg: Config, d_from: date, d_to: date) -> None:
    """Pull daily + 5-min + 1-min bars for the whole universe into the cache."""
    client = FMPClient(cfg.fmp_api_key)
    store = Store(cfg)
    # daily context needs lookback for ATR14 / GK-vol z-score (60d) warmup
    daily_from = d_from - timedelta(days=140)
    try:
        for sym in cfg.universe:
            log.info("[%s] daily %s -> %s", sym, daily_from, d_to)
            store.save_daily(sym, client.daily(sym, daily_from, d_to))
            for tf in (cfg.raw["data"]["detect_tf"], cfg.raw["data"]["refine_tf"]):
                log.info("[%s] %s %s -> %s", sym, tf, d_from, d_to)
                bars = client.intraday(sym, tf, d_from, d_to)
                if bars.height == 0:
                    log.warning("[%s] no %s bars returned — check FMP plan depth", sym, tf)
                store.save_bars(tf, sym, bars)
    finally:
        client.close()
