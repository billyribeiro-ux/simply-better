"""Write run artifacts as the dashboard's static JSON contract."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import Config

log = logging.getLogger("engine.export")


def write(cfg: Config, artifacts: dict) -> Path:
    out = cfg.export_dir
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": artifacts["run_id"],
        "run_type": artifacts.get("run_type"),
        "from": artifacts["from"],
        "to": artifacts["to"],
        "universe": artifacts["universe"],
        "kpis": artifacts["summary"],
        "per_symbol": artifacts["per_symbol"],
        "per_setup": artifacts["per_setup"],
        "feature_importance": artifacts["feature_importance"],
        "folds": artifacts["folds"],
        "diagnostics": artifacts.get("diagnostics", {"folds": [], "auc_mean_test": None}),
        "anchor_recovery": artifacts.get("anchor_recovery", []),
    }
    _dump(out / "summary.json", summary)
    _dump(out / "equity.json", artifacts["equity"])
    _dump(out / "signals.json", {"signals": artifacts["signals"]})
    _dump(out / "geometry.json", {"rows": artifacts["geometry"]})
    _dump(out / "attribution.json", artifacts["attribution"])
    log.info("exported dashboard data -> %s", out)
    return out


def _dump(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
