"""Volatility helpers used by risk sizing and stop calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder-style ATR on OHLC with lowercase columns."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def latest_atr_pct(df: pd.DataFrame | None, window: int = 14) -> float:
    """Latest ATR expressed as a fraction of the last close. NaN if insufficient data."""
    if df is None or df.empty or len(df) < window + 1:
        return float("nan")
    series = atr(df, window)
    if series.empty:
        return float("nan")
    a = float(series.iloc[-1])
    price = float(df["close"].iloc[-1])
    if not price or np.isnan(price) or np.isnan(a):
        return float("nan")
    return a / price


def gap_pct(df: pd.DataFrame | None) -> float:
    """Latest overnight gap: (today_open - prev_close) / prev_close, 0 if unknown."""
    if df is None or df.empty or len(df) < 2:
        return 0.0
    prev_close = float(df["close"].iloc[-2])
    today_open = float(df["open"].iloc[-1])
    if not prev_close or np.isnan(prev_close) or np.isnan(today_open):
        return 0.0
    return (today_open - prev_close) / prev_close
