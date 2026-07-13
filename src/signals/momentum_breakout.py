"""Sparse high-momentum breakout signal."""
from __future__ import annotations

from math import sqrt

import pandas as pd

from src.config import Config


def momentum_breakout_scores(
    cfg: Config,
    history: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Return 1.0 for the top qualifying breakout name, else 0.0.

    The rule uses only data present in each symbol's supplied history. Callers
    should pass prior-bar history in backtests to avoid lookahead.
    """
    mb_cfg = cfg.get("strategies", "momentum_breakout", default={}) or {}
    if not bool(mb_cfg.get("enabled", False)):
        return {}

    benchmark = str(mb_cfg.get("benchmark_symbol", "QQQ")).upper()
    benchmark_hist = history.get(benchmark)
    if benchmark_hist is None or benchmark_hist.empty:
        return {sym: 0.0 for sym in history}

    lookback = int(mb_cfg.get("lookback_days", 252))
    min_return = float(mb_cfg.get("min_return", 5.0))
    sma_window = int(mb_cfg.get("sma_window", 100))
    vol_window = int(mb_cfg.get("volatility_window", 21))
    max_annualized_vol = float(mb_cfg.get("max_annualized_vol", 0.60))
    benchmark_sma_window = int(mb_cfg.get("benchmark_sma_window", 200))
    top_n = max(1, int(mb_cfg.get("top_n", 1)))
    excluded = {str(s).upper() for s in mb_cfg.get("exclude_symbols", [])}

    benchmark_close = benchmark_hist["close"].dropna()
    if len(benchmark_close) < benchmark_sma_window:
        return {sym: 0.0 for sym in history}
    benchmark_sma = benchmark_close.rolling(benchmark_sma_window).mean().iloc[-1]
    if benchmark_sma != benchmark_sma or benchmark_close.iloc[-1] <= benchmark_sma:
        return {sym: 0.0 for sym in history}

    candidates: list[tuple[str, float]] = []
    min_bars = max(lookback + 1, sma_window, vol_window + 1)
    for sym, df in history.items():
        symbol = sym.upper()
        if symbol in excluded:
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = df["close"].dropna()
        if len(close) < min_bars:
            continue
        start = close.iloc[-lookback - 1]
        end = close.iloc[-1]
        if start <= 0:
            continue
        trailing_return = float(end / start - 1.0)
        if trailing_return <= min_return:
            continue
        sma = close.rolling(sma_window).mean().iloc[-1]
        if sma != sma or end <= sma:
            continue
        annualized_vol = float(close.pct_change().rolling(vol_window).std().iloc[-1] * sqrt(252))
        if annualized_vol != annualized_vol or annualized_vol >= max_annualized_vol:
            continue
        candidates.append((sym, trailing_return))

    winners = {
        sym
        for sym, _ in sorted(candidates, key=lambda item: item[1], reverse=True)[:top_n]
    }
    return {sym: (1.0 if sym in winners else 0.0) for sym in history}
