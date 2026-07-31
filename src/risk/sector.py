"""Dynamic sector-risk measurements derived from the equity universe.

Static position-count caps prevent concentration after orders are selected.
This module adds a separate, point-in-time sizing overlay based on sector
breadth and realized volatility.  It never increases a proposed position.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from src.data.sectors import sector_for


@dataclass(frozen=True)
class SectorRisk:
    sector: str
    members: int
    breadth: float
    annualized_volatility: float
    multiplier: float


def calculate_sector_risk(
    history: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> dict[str, SectorRisk]:
    """Return sector sizing multipliers using only each history's latest bars."""
    if not bool(config.get("enabled", False)):
        return {}

    sma_window = max(2, int(config.get("sma_window", 50)))
    vol_window = max(2, int(config.get("volatility_window", 20)))
    min_members = max(1, int(config.get("min_members", 3)))
    min_breadth = float(config.get("min_breadth", 0.40))
    weak_multiplier = float(config.get("weak_breadth_multiplier", 0.50))
    max_vol = float(config.get("max_annualized_vol", 0.50))
    min_multiplier = float(config.get("min_multiplier", 0.25))

    observations: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    required = max(sma_window, vol_window + 1)
    for symbol, frame in history.items():
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if len(close) < required or close.iloc[-1] <= 0:
            continue
        sma = float(close.iloc[-sma_window:].mean())
        returns = close.pct_change().dropna().iloc[-vol_window:]
        if len(returns) < vol_window or not np.isfinite(sma):
            continue
        annualized_vol = float(returns.std(ddof=1) * sqrt(252))
        if not np.isfinite(annualized_vol):
            continue
        observations[sector_for(symbol)].append((bool(close.iloc[-1] >= sma), annualized_vol))

    result: dict[str, SectorRisk] = {}
    for sector, values in observations.items():
        if len(values) < min_members:
            continue
        breadth = sum(above_sma for above_sma, _ in values) / len(values)
        sector_vol = float(np.median([vol for _, vol in values]))
        breadth_multiplier = weak_multiplier if breadth < min_breadth else 1.0
        vol_multiplier = 1.0 if max_vol <= 0 or sector_vol <= max_vol else max_vol / sector_vol
        multiplier = max(min_multiplier, min(1.0, breadth_multiplier, vol_multiplier))
        result[sector] = SectorRisk(
            sector=sector,
            members=len(values),
            breadth=float(breadth),
            annualized_volatility=sector_vol,
            multiplier=float(multiplier),
        )
    return result
