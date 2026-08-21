"""Pure return and performance calculations for a close-to-next-open strategy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class OvernightBacktestResult:
    daily_returns: pd.Series
    equity_curve: pd.Series
    ending_value: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    normalized = bars.rename(columns=lambda value: str(value).lower())
    if not {"open", "close"}.issubset(normalized.columns):
        raise ValueError("bars must contain open and close columns")
    prices = normalized[["open", "close"]].astype(float).sort_index()
    if prices.empty or prices.isna().any().any():
        raise ValueError("open and close prices must be present")
    if (prices <= 0.0).any().any():
        raise ValueError("open and close prices must be positive")
    return prices


def calculate_session_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Split each adjusted close-to-close return into overnight and intraday legs."""
    prices = _validated_bars(bars)
    returns = pd.DataFrame(index=prices.index)
    returns["overnight"] = prices["open"] / prices["close"].shift(1) - 1.0
    returns["intraday"] = prices["close"] / prices["open"] - 1.0
    returns["close_to_close"] = prices["close"].pct_change(fill_method=None)
    return returns


def backtest_overnight(
    bars: pd.DataFrame,
    *,
    cost_bps_per_side: float = 0.0,
) -> OvernightBacktestResult:
    """Compound daily overnight returns after a cost on the buy and the sell."""
    if not 0.0 <= cost_bps_per_side <= 10_000.0:
        raise ValueError("cost_bps_per_side must be between 0 and 10,000")

    gross_returns = calculate_session_returns(bars)["overnight"].dropna()
    if gross_returns.empty:
        raise ValueError("at least two price rows are required")

    side_factor = 1.0 - cost_bps_per_side / 10_000.0
    daily_returns = (1.0 + gross_returns) * side_factor**2 - 1.0
    equity_curve = (1.0 + daily_returns).cumprod()
    trades = len(daily_returns)
    ending_value = float(equity_curve.iloc[-1])
    total_return = ending_value - 1.0
    cagr = ending_value ** (TRADING_DAYS_PER_YEAR / trades) - 1.0
    annual_volatility = float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    daily_std = float(daily_returns.std(ddof=1))
    sharpe = (
        float(daily_returns.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std > 0.0
        else float("nan")
    )
    drawdown = pd.concat([pd.Series([1.0]), equity_curve], ignore_index=True)
    max_drawdown = float((drawdown / drawdown.cummax() - 1.0).min())

    return OvernightBacktestResult(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        ending_value=ending_value,
        total_return=total_return,
        cagr=cagr,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=float((daily_returns > 0.0).mean()),
        trades=trades,
    )
