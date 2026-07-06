"""Typed configuration loading. Single source of truth: config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    # ---- convenience accessors -------------------------------------------
    @property
    def universe(self) -> list[str]:
        return list(self.raw["universe"])

    @property
    def cache_dir(self) -> Path:
        return self.root / self.raw["data"]["cache_dir"]

    @property
    def db_path(self) -> Path:
        return self.root / self.raw["data"]["db_path"]

    @property
    def session(self) -> dict[str, str]:
        return self.raw["session"]

    @property
    def setups(self) -> dict[str, Any]:
        return self.raw["setups"]

    @property
    def entry(self) -> dict[str, Any]:
        return self.raw["entry"]

    @property
    def geometry(self) -> dict[str, Any]:
        return self.raw["geometry"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def learning(self) -> dict[str, Any]:
        return self.raw["learning"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw["execution"]

    @property
    def export_dir(self) -> Path:
        return self.root / self.raw["export"]["out_dir"]

    @property
    def scan_grid(self) -> np.ndarray:
        g = self.geometry["scan_grid"]
        return np.round(
            np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 2
        )

    @property
    def fmp_api_key(self) -> str:
        key = os.environ.get("FMP_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "FMP_API_KEY is not set. Export it before running: "
                "export FMP_API_KEY=your_key"
            )
        return key


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for section in ("universe", "data", "session", "setups", "geometry",
                    "model", "learning", "execution", "export"):
        if section not in raw:
            raise ValueError(f"config.yaml missing required section: {section}")
    return Config(raw=raw, root=p.parent)
