"""Configuration loader: merges config.yaml with .env secrets."""
from __future__ import annotations

import os
from math import isfinite
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]
    api_key: str
    api_secret: str
    is_live: bool

    def get(self, *keys, default=None):
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: str | Path = "config.yaml", require_secrets: bool = True) -> Config:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    _validate_volatility_targeting(raw)

    mode = (raw.get("mode") or "paper").lower()
    is_live = mode == "live"

    if is_live:
        key = os.getenv("ALPACA_LIVE_API_KEY", "")
        secret = os.getenv("ALPACA_LIVE_API_SECRET", "")
        if require_secrets and (not key or not secret):
            raise RuntimeError(
                "mode=live in config.yaml but ALPACA_LIVE_API_KEY/SECRET not set in .env. "
                "Refusing to start."
            )
    else:
        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_API_SECRET", "")
        if require_secrets and (not key or not secret):
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_API_SECRET missing in .env. "
                "Get free paper keys at https://app.alpaca.markets/paper/dashboard/overview"
            )

    return Config(raw=raw, api_key=key, api_secret=secret, is_live=is_live)


def _validate_volatility_targeting(raw: dict[str, Any]) -> None:
    risk = raw.get("risk", {}) or {}
    vol = risk.get("volatility_targeting", {}) or {}
    if not vol:
        return

    benchmark = str(vol.get("benchmark_symbol", "")).strip()
    lookback = vol.get("lookback_days")
    if not benchmark:
        raise ValueError("risk.volatility_targeting.benchmark_symbol must be nonempty")
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 10:
        raise ValueError("risk.volatility_targeting.lookback_days must be an integer >= 10")

    names = (
        "target_annualized_vol",
        "risk_on_min_gross_exposure",
        "risk_on_max_gross_exposure",
        "realized_vol_floor",
        "realized_vol_ceiling",
        "rebalance_band_pct",
        "min_rebalance_dollars",
    )
    values: dict[str, float] = {}
    for name in names:
        try:
            value = float(vol.get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"risk.volatility_targeting.{name} must be finite") from exc
        if not isfinite(value):
            raise ValueError(f"risk.volatility_targeting.{name} must be finite")
        values[name] = value

    if values["target_annualized_vol"] <= 0:
        raise ValueError("target_annualized_vol must be > 0")
    if values["realized_vol_floor"] <= 0 or values["realized_vol_ceiling"] <= 0:
        raise ValueError("realized volatility bounds must be > 0")
    if values["realized_vol_floor"] > values["realized_vol_ceiling"]:
        raise ValueError("realized_vol_floor must be <= realized_vol_ceiling")
    for name in ("risk_on_min_gross_exposure", "risk_on_max_gross_exposure", "rebalance_band_pct"):
        if not 0 <= values[name] <= 1:
            raise ValueError(f"{name} must be within [0, 1]")
    if values["risk_on_min_gross_exposure"] > values["risk_on_max_gross_exposure"]:
        raise ValueError("risk_on_min_gross_exposure must be <= risk_on_max_gross_exposure")
    if values["risk_on_max_gross_exposure"] > 0.95:
        raise ValueError("risk_on_max_gross_exposure cannot exceed the 95% hard ceiling")
    normal = float(risk.get("max_gross_exposure", 0.80))
    if not values["risk_on_min_gross_exposure"] <= normal <= values["risk_on_max_gross_exposure"]:
        raise ValueError("risk.max_gross_exposure must be inside the volatility-targeting risk-on range")
    if values["min_rebalance_dollars"] < 0:
        raise ValueError("min_rebalance_dollars must be >= 0")
