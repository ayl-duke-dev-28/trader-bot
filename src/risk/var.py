"""Historical portfolio VaR, expected shortfall, and pre-trade buy gating."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, TypeVar

import numpy as np
import pandas as pd


class IntentLike(Protocol):
    symbol: str
    side: str
    target_dollars: float


TIntent = TypeVar("TIntent", bound=IntentLike)


@dataclass(frozen=True)
class HistoricalRiskEstimate:
    var_pct: float
    expected_shortfall_pct: float
    observations: int
    confidence: float


@dataclass
class VarGateDiagnostics:
    enabled: bool
    accepted_buys: int = 0
    blocked_buys: int = 0
    blocked_symbols: list[str] = field(default_factory=list)
    blocked_estimates: dict[str, HistoricalRiskEstimate | None] = field(default_factory=dict)
    latest_estimate: HistoricalRiskEstimate | None = None


def estimate_historical_risk(
    history: dict[str, pd.DataFrame],
    exposures: dict[str, float],
    *,
    equity: float,
    confidence: float,
    lookback_days: int,
    min_observations: int,
) -> HistoricalRiskEstimate | None:
    """Estimate one-day portfolio loss quantile and average loss beyond it."""
    if not np.isfinite(equity) or equity <= 0:
        return None
    if not 0.5 < confidence < 1.0:
        return None
    active = {
        str(symbol).upper(): float(value)
        for symbol, value in exposures.items()
        if np.isfinite(value) and abs(float(value)) > 1e-9
    }
    if not active:
        return HistoricalRiskEstimate(0.0, 0.0, 0, confidence)

    closes: dict[str, pd.Series] = {}
    for symbol in active:
        frame = history.get(symbol)
        if frame is None or frame.empty or "close" not in frame.columns:
            return None
        close = pd.to_numeric(frame["close"], errors="coerce")
        closes[symbol] = close.where(close > 0)

    close_frame = pd.DataFrame(closes).sort_index().tail(max(2, int(lookback_days) + 1))
    returns = close_frame.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="any")
    if len(returns) < max(1, int(min_observations)):
        return None

    weights = pd.Series(active, dtype=float) / float(equity)
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1).to_numpy(dtype=float)
    losses = -portfolio_returns
    var_pct = max(0.0, float(np.quantile(losses, confidence)))
    tail = losses[losses >= var_pct]
    expected_shortfall_pct = max(0.0, float(tail.mean())) if len(tail) else var_pct
    return HistoricalRiskEstimate(
        var_pct=var_pct,
        expected_shortfall_pct=expected_shortfall_pct,
        observations=len(returns),
        confidence=confidence,
    )


def filter_buy_intents_by_var(
    intents: Iterable[TIntent],
    *,
    current_exposures: dict[str, float],
    history: dict[str, pd.DataFrame],
    equity: float,
    config: dict,
) -> tuple[list[TIntent], VarGateDiagnostics]:
    """Keep sells and only accept buys whose projected portfolio risk fits limits."""
    intent_list = list(intents)
    enabled = bool(config.get("enabled", False))
    diagnostics = VarGateDiagnostics(enabled=enabled)
    if not enabled:
        return intent_list, diagnostics

    exposures = {
        str(symbol).upper(): max(0.0, float(value))
        for symbol, value in current_exposures.items()
        if np.isfinite(value) and float(value) > 0
    }
    accepted: list[TIntent] = []

    # Sells reduce projected exposure and must never be blocked by a VaR gate.
    for intent in intent_list:
        if intent.side.lower() != "sell":
            continue
        accepted.append(intent)
        symbol = intent.symbol.upper()
        exposures[symbol] = max(0.0, exposures.get(symbol, 0.0) - intent.target_dollars)

    confidence = float(config.get("confidence", 0.99))
    lookback_days = int(config.get("lookback_days", 252))
    min_observations = int(config.get("min_observations", 60))
    max_var_pct = float(config.get("max_var_pct", 0.03))
    max_es_pct = float(config.get("max_expected_shortfall_pct", 0.04))
    fail_closed = bool(config.get("fail_closed", True))

    for intent in intent_list:
        if intent.side.lower() != "buy":
            continue
        symbol = intent.symbol.upper()
        projected = dict(exposures)
        projected[symbol] = projected.get(symbol, 0.0) + max(0.0, intent.target_dollars)
        estimate = estimate_historical_risk(
            history,
            projected,
            equity=equity,
            confidence=confidence,
            lookback_days=lookback_days,
            min_observations=min_observations,
        )
        diagnostics.latest_estimate = estimate
        insufficient = estimate is None
        over_limit = bool(
            estimate is not None
            and (
                estimate.var_pct > max_var_pct + 1e-12
                or estimate.expected_shortfall_pct > max_es_pct + 1e-12
            )
        )
        if over_limit or (insufficient and fail_closed):
            diagnostics.blocked_buys += 1
            diagnostics.blocked_symbols.append(symbol)
            diagnostics.blocked_estimates[symbol] = estimate
            continue
        accepted.append(intent)
        exposures = projected
        diagnostics.accepted_buys += 1

    return accepted, diagnostics
