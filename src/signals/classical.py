"""Classical technical signals combined into a single score in [-1, 1].

A positive score means "buy bias", negative means "sell/short bias".
Each sub-signal is normalized to [-1, 1] and averaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator


def _safe(score: float) -> float:
    if score != score:  # NaN
        return 0.0
    return float(np.clip(score, -1.0, 1.0))


def momentum_score(close: pd.Series, lookback: int = 60) -> float:
    if len(close) < lookback + 1:
        return 0.0
    ret = close.iloc[-1] / close.iloc[-lookback] - 1.0
    # squash through tanh so a 20% return ≈ 0.76
    return _safe(np.tanh(ret * 4))


def mean_reversion_score(close: pd.Series, lookback: int = 20) -> float:
    if len(close) < lookback + 1:
        return 0.0
    ma = close.rolling(lookback).mean().iloc[-1]
    std = close.rolling(lookback).std().iloc[-1]
    if std == 0 or std != std:
        return 0.0
    z = (close.iloc[-1] - ma) / std
    # negative z -> oversold -> buy; positive z -> overbought -> sell
    return _safe(-np.tanh(z / 2))


def ma_cross_score(close: pd.Series, fast: int = 20, slow: int = 50) -> float:
    if len(close) < slow + 1:
        return 0.0
    f = SMAIndicator(close, window=fast).sma_indicator().iloc[-1]
    s = SMAIndicator(close, window=slow).sma_indicator().iloc[-1]
    if s == 0 or s != s:
        return 0.0
    spread = (f - s) / s
    return _safe(np.tanh(spread * 20))


def rsi_score(close: pd.Series, window: int = 14) -> float:
    if len(close) < window + 1:
        return 0.0
    rsi = RSIIndicator(close, window=window).rsi().iloc[-1]
    if rsi != rsi:
        return 0.0
    # 30 → +1 (oversold buy), 70 → -1 (overbought sell), 50 → 0
    return _safe((50 - rsi) / 20)


def macd_score(close: pd.Series) -> float:
    if len(close) < 35:
        return 0.0
    macd = MACD(close)
    hist = macd.macd_diff().iloc[-1]
    last_price = close.iloc[-1]
    if last_price == 0:
        return 0.0
    return _safe(np.tanh((hist / last_price) * 200))


def classical_signal(df: pd.DataFrame) -> float:
    """Combine the five signals into a single score in [-1, 1]."""
    if df is None or df.empty or "close" not in df.columns:
        return 0.0
    close = df["close"]
    parts = [
        momentum_score(close),
        mean_reversion_score(close),
        ma_cross_score(close),
        rsi_score(close),
        macd_score(close),
    ]
    return _safe(float(np.mean(parts)))
