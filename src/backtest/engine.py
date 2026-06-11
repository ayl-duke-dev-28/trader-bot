"""Walk-forward backtester.

Each bar:
  - compute composite signal per symbol from history up to (but not including) that bar
  - target dollar weight per symbol = clip(score, 0, 1) * max_position_pct,
    renormalized to <= max_gross_exposure
  - rebalance daily at close
  - apply a flat commission/slippage in bps

Returns a metrics dict and an equity curve.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config
from src.signals.classical import classical_signal
from src.signals.ml import load_model, ml_signal

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    cagr: float
    sharpe: float
    max_drawdown: float
    total_return: float
    trades: int


def _composite_score(cfg: Config, df: pd.DataFrame, bundle) -> float:
    w_cls = float(cfg.get("strategies", "classical", "weight", default=0.5))
    w_ml = float(cfg.get("strategies", "ml", "weight", default=0.5))
    cls = classical_signal(df) if cfg.get("strategies", "classical", "enabled", default=True) else 0.0
    ml = ml_signal(cfg, df, bundle=bundle) if (cfg.get("strategies", "ml", "enabled", default=True) and bundle) else 0.0
    total_w = max(1e-9, w_cls + w_ml)
    return float((w_cls * cls + w_ml * ml) / total_w)


def backtest(
    cfg: Config,
    history: dict[str, pd.DataFrame],
    start_capital: float = 100_000.0,
    rebalance_every: int = 5,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Backtest over the intersection of dates across all symbols."""
    if not history:
        raise ValueError("history is empty")

    # Align on a common daily index
    closes = pd.DataFrame({s: df["close"] for s, df in history.items()}).dropna(how="all")
    closes = closes.ffill().dropna()
    if closes.empty:
        raise ValueError("no overlapping price history")

    bundle = load_model(cfg) if cfg.get("strategies", "ml", "enabled", default=True) else None
    max_pos = float(cfg.get("risk", "max_position_pct", default=0.05))
    max_gross = float(cfg.get("risk", "max_gross_exposure", default=0.80))

    equity = start_capital
    weights = pd.Series(0.0, index=closes.columns)
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    trade_count = 0

    rebalance_dates = closes.index[::rebalance_every]
    rebalance_set = set(rebalance_dates)

    # We need at least ~60 bars of warmup
    warmup = 60
    if len(closes) <= warmup:
        raise ValueError("not enough history for backtest")

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        # Apply yesterday's weights to today's return
        if i > warmup:
            day_ret = (closes.iloc[i] / closes.iloc[i - 1] - 1.0)
            port_ret = float((weights * day_ret).sum())
            equity *= 1.0 + port_ret

        if date in rebalance_set:
            scores = {}
            for sym in closes.columns:
                sym_hist = history[sym].loc[:date].iloc[:-1]  # exclude today
                scores[sym] = _composite_score(cfg, sym_hist, bundle)
            # Long-only sizing
            raw = {s: max(0.0, sc) for s, sc in scores.items()}
            target = pd.Series({s: min(max_pos, raw[s] * max_pos / max(1e-9, max(raw.values()))) for s in raw})
            gross = target.sum()
            if gross > max_gross:
                target *= max_gross / gross
            # Charge cost on weight changes
            turnover = (target - weights).abs().sum()
            equity *= 1.0 - turnover * (cost_bps / 10_000.0)
            trade_count += int((target - weights).abs().sum() > 0)
            weights = target

        equity_curve.append((date, equity))

    curve = pd.Series({d: v for d, v in equity_curve})
    rets = curve.pct_change().dropna()
    total_return = curve.iloc[-1] / start_capital - 1.0
    years = max(1e-9, (curve.index[-1] - curve.index[0]).days / 365.25)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0.0
    rolling_max = curve.cummax()
    drawdown = (curve / rolling_max - 1.0).min()

    return BacktestResult(
        equity_curve=curve,
        cagr=float(cagr),
        sharpe=sharpe,
        max_drawdown=float(drawdown),
        total_return=float(total_return),
        trades=trade_count,
    )
