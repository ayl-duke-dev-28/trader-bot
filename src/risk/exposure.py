"""Pure market-regime and volatility-targeted exposure policy.

The policy is deliberately independent of broker/backtest adapters so identical
benchmark history produces identical exposure decisions in both paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    UNKNOWN = "UNKNOWN"


class ExposureReason(str, Enum):
    DISABLED = "DISABLED"
    VOL_SCALED = "VOL_SCALED"
    CLAMPED_MIN = "CLAMPED_MIN"
    CLAMPED_MAX = "CLAMPED_MAX"
    RISK_OFF = "RISK_OFF"
    FALLBACK_HISTORY = "FALLBACK_HISTORY"
    FALLBACK_NONFINITE = "FALLBACK_NONFINITE"
    FALLBACK_REGIME = "FALLBACK_REGIME"


@dataclass(frozen=True)
class ExposureDecision:
    gross_target: float
    regime: MarketRegime
    reason: ExposureReason
    realized_vol: float | None


def market_regime(
    benchmark_history: pd.DataFrame | None,
    sma_window: int,
) -> MarketRegime:
    """Classify a benchmark using only the bars supplied by the caller."""
    close = _valid_closes(benchmark_history)
    if close is None or len(close) < sma_window:
        return MarketRegime.UNKNOWN
    sma = float(close.tail(sma_window).mean())
    last = float(close.iloc[-1])
    if not isfinite(sma):
        return MarketRegime.UNKNOWN
    return MarketRegime.RISK_ON if last >= sma else MarketRegime.RISK_OFF


def exposure_decision(
    benchmark_history: pd.DataFrame | None,
    *,
    enabled: bool,
    normal_max_gross: float,
    risk_off_max_gross: float,
    current_gross: float,
    sma_window: int,
    lookback_days: int,
    target_annualized_vol: float,
    risk_on_min_gross: float,
    risk_on_max_gross: float,
    realized_vol_floor: float,
    realized_vol_ceiling: float,
) -> ExposureDecision:
    """Return the allowed gross exposure for the supplied completed history.

    Invalid history never creates fresh buy capacity when volatility targeting
    is enabled. With the feature disabled, the legacy regime behavior is kept.
    """
    regime = market_regime(benchmark_history, sma_window)
    if regime is MarketRegime.RISK_OFF:
        return ExposureDecision(
            gross_target=min(normal_max_gross, risk_off_max_gross),
            regime=regime,
            reason=ExposureReason.RISK_OFF,
            realized_vol=None,
        )

    if not enabled:
        return ExposureDecision(
            gross_target=normal_max_gross,
            regime=regime,
            reason=ExposureReason.DISABLED,
            realized_vol=None,
        )

    fallback = max(0.0, min(normal_max_gross, current_gross))
    if regime is MarketRegime.UNKNOWN:
        return ExposureDecision(fallback, regime, ExposureReason.FALLBACK_REGIME, None)

    close = _valid_closes(benchmark_history)
    if close is None or len(close) < lookback_days + 1:
        return ExposureDecision(fallback, regime, ExposureReason.FALLBACK_HISTORY, None)

    returns = close.tail(lookback_days + 1).pct_change().dropna()
    realized_vol = float(returns.std(ddof=1) * sqrt(252))
    if len(returns) != lookback_days or not isfinite(realized_vol) or realized_vol <= 0:
        return ExposureDecision(fallback, regime, ExposureReason.FALLBACK_NONFINITE, None)

    bounded_vol = float(np.clip(realized_vol, realized_vol_floor, realized_vol_ceiling))
    raw_target = normal_max_gross * target_annualized_vol / bounded_vol
    gross_target = float(np.clip(raw_target, risk_on_min_gross, risk_on_max_gross))
    if raw_target < risk_on_min_gross:
        reason = ExposureReason.CLAMPED_MIN
    elif raw_target > risk_on_max_gross:
        reason = ExposureReason.CLAMPED_MAX
    else:
        reason = ExposureReason.VOL_SCALED
    return ExposureDecision(gross_target, regime, reason, realized_vol)


def _valid_closes(history: pd.DataFrame | None) -> pd.Series | None:
    if history is None or history.empty or "close" not in history.columns:
        return None
    close = pd.to_numeric(history["close"], errors="coerce")
    if close.empty or close.isna().any() or (~np.isfinite(close)).any() or (close <= 0).any():
        return None
    return close
