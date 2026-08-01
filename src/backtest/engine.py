"""Backtest entry points.

The public ``backtest`` function replays the same daily decision path used by
the live trader: signal generation, benchmark core sleeve, regime filter,
relative strength, sector caps, gap skips, stop/trailing exits, cooldowns, and
whole/fractional-share sizing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type, datetime
from typing import Any

import pandas as pd

from src.backtest.simulator import simulate_current_bot
from src.config import Config
from src.signals.ml import train_model_bundle

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    cagr: float
    sharpe: float
    max_drawdown: float
    total_return: float
    trades: int
    trades_log: pd.DataFrame | None = None
    summary: dict[str, float | int | str] | None = None
    equity_diagnostics: pd.DataFrame | None = None


def backtest(
    cfg: Config,
    history: dict[str, pd.DataFrame],
    start_capital: float = 100_000.0,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    cost_bps: float = 5.0,
    earnings_calendar: dict[str, list[date_type | datetime | pd.Timestamp | str]] | None = None,
    walk_forward: bool = True,
    train_window_days: int | None = None,
    test_window_days: int | None = None,
    macro_cycles: pd.DataFrame | None = None,
    macro_cycle_config: dict[str, Any] | None = None,
) -> BacktestResult:
    """Replay the live bot's current decision path over historical daily bars."""
    if not history:
        raise ValueError("history is empty")

    closes = pd.DataFrame({s: df["close"] for s, df in history.items()}).sort_index()
    closes = closes.dropna(how="all").ffill()
    if closes.empty:
        raise ValueError("no price history")

    if start_date is None:
        warmup_days = int(cfg.get("backtest", "warmup_days", default=280))
        first_date = pd.Timestamp(closes.index.min())
        start_date = first_date + pd.Timedelta(days=warmup_days)

    model_provider = None
    walk_windows: list[dict[str, str | int]] = []
    if walk_forward and cfg.get("strategies", "ml", "enabled", default=True):
        train_days = train_window_days or int(cfg.get("backtest", "train_window_days", default=756))
        test_days = test_window_days or int(cfg.get("backtest", "test_window_days", default=63))
        model_provider, walk_windows = _walk_forward_model_provider(
            cfg=cfg,
            history=history,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(closes.index.max()),
            train_window_days=train_days,
            test_window_days=test_days,
        )

    result = simulate_current_bot(
        cfg,
        history,
        start_date=pd.Timestamp(start_date),
        end_date=None if end_date is None else pd.Timestamp(end_date),
        start_capital=start_capital,
        cost_bps=cost_bps,
        earnings_calendar=earnings_calendar,
        model_provider=model_provider,
        macro_cycles=macro_cycles,
        macro_cycle_config=macro_cycle_config,
    )
    if walk_windows:
        result.summary["walk_forward_windows"] = len(walk_windows)
        result.summary["walk_forward_train_window_days"] = int(train_window_days or cfg.get("backtest", "train_window_days", default=756))
        result.summary["walk_forward_test_window_days"] = int(test_window_days or cfg.get("backtest", "test_window_days", default=63))
    equity_curve = result.equity_curve.set_index(pd.to_datetime(result.equity_curve["date"]))["equity"]
    return BacktestResult(
        equity_curve=equity_curve,
        cagr=float(result.summary["cagr"]),
        sharpe=float(result.summary["sharpe"]),
        max_drawdown=float(result.summary["max_drawdown"]),
        total_return=float(result.summary["total_return"]),
        trades=int(result.summary["trades"]),
        trades_log=result.trades,
        summary=result.summary,
        equity_diagnostics=result.equity_curve,
    )


def _walk_forward_model_provider(
    cfg: Config,
    history: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_window_days: int,
    test_window_days: int,
):
    """Train ML bundles on rolling prior windows and serve them by test date."""
    horizon = int(cfg.get("strategies", "ml", "horizon_days", default=5))
    window_starts = pd.date_range(start=start_date, end=end_date, freq=f"{test_window_days}D")
    if len(window_starts) == 0 or window_starts[0] != start_date:
        window_starts = pd.DatetimeIndex([start_date, *window_starts])

    windows: list[tuple[pd.Timestamp, pd.Timestamp, dict[str, Any] | None]] = []
    metadata: list[dict[str, str | int]] = []
    for idx, test_start in enumerate(window_starts):
        next_start = window_starts[idx + 1] if idx + 1 < len(window_starts) else end_date + pd.Timedelta(days=1)
        test_end = min(end_date, next_start - pd.Timedelta(days=1))
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=train_window_days)
        train_hist = {
            sym: df.loc[(df.index >= train_start) & (df.index <= train_end)]
            for sym, df in history.items()
        }
        train_hist = {sym: df for sym, df in train_hist.items() if not df.empty}
        log.info(
            "walk-forward train %s to %s; test %s to %s on %d symbols",
            train_start.date(),
            train_end.date(),
            test_start.date(),
            test_end.date(),
            len(train_hist),
        )
        bundle = train_model_bundle(train_hist, horizon_days=horizon) if train_hist else None
        windows.append((pd.Timestamp(test_start), pd.Timestamp(test_end), bundle))
        metadata.append({
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "symbols": len(train_hist),
        })

    def provider(date: pd.Timestamp) -> dict[str, Any] | None:
        date = pd.Timestamp(date)
        for test_start, test_end, bundle in windows:
            if test_start <= date <= test_end:
                return bundle
        return windows[-1][2] if windows else None

    return provider, metadata
