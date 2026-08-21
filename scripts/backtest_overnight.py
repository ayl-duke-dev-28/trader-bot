#!/usr/bin/env python3
"""Backtest buying adjusted market closes and selling the next market open."""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.overnight import backtest_overnight, calculate_session_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM", "DIA", "VTI"])
    parser.add_argument("--start", default=(date.today() - timedelta(days=round(365.25 * 20))).isoformat())
    parser.add_argument("--end", default=(date.today() + timedelta(days=1)).isoformat())
    parser.add_argument("--costs", nargs="+", type=float, default=[0.0, 1.0, 2.0, 5.0])
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/backtests/overnight_20y"))
    return parser.parse_args()


def generic_metrics(returns: pd.Series) -> dict[str, float | int]:
    values = returns.dropna()
    equity = (1.0 + values).cumprod()
    ending = float(equity.iloc[-1])
    std = float(values.std(ddof=1))
    with_initial = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    return {
        "trades": len(values),
        "total_return": ending - 1.0,
        "cagr": ending ** (252.0 / len(values)) - 1.0,
        "annual_volatility": std * np.sqrt(252.0),
        "sharpe": float(values.mean() / std * np.sqrt(252.0)) if std > 0.0 else np.nan,
        "max_drawdown": float((with_initial / with_initial.cummax() - 1.0).min()),
        "win_rate": float((values > 0.0).mean()),
        "ending_multiple": ending,
    }


def download_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    bars = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    if bars.empty:
        raise RuntimeError(f"no Yahoo Finance data returned for {symbol}")
    return bars.rename(columns=str.lower)


def result_row(
    symbol: str,
    strategy: str,
    cost_bps_per_side: float | None,
    metrics: dict[str, float | int],
    initial_capital: float,
) -> dict[str, float | int | str | None]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "cost_bps_per_side": cost_bps_per_side,
        **metrics,
        "ending_value": float(metrics["ending_multiple"]) * initial_capital,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in ["total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate"]:
        display[column] = display[column].map(lambda value: f"{value:.1%}")
    display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")
    display["ending_value"] = display["ending_value"].map(lambda value: f"${value:,.0f}")
    columns = [
        "symbol", "strategy", "cost_bps_per_side", "total_return", "cagr",
        "annual_volatility", "sharpe", "max_drawdown", "ending_value", "trades",
    ]
    table = display[columns].fillna("-").astype(str)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | int | str | None]] = []
    equity_curves: dict[str, pd.Series] = {}

    for symbol in args.symbols:
        bars = download_bars(symbol, args.start, args.end)
        session_returns = calculate_session_returns(bars)
        rows.append(result_row(symbol, "buy_and_hold", None, generic_metrics(session_returns["close_to_close"]), args.initial_capital))
        rows.append(result_row(symbol, "intraday", None, generic_metrics(session_returns["intraday"]), args.initial_capital))
        for cost in args.costs:
            result = backtest_overnight(bars, cost_bps_per_side=cost)
            metrics = {
                "trades": result.trades,
                "total_return": result.total_return,
                "cagr": result.cagr,
                "annual_volatility": result.annual_volatility,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "ending_multiple": result.ending_value,
            }
            rows.append(result_row(symbol, "overnight", cost, metrics, args.initial_capital))
            equity_curves[f"{symbol}_overnight_{cost:g}bps"] = result.equity_curve * args.initial_capital

    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(equity_curves).to_csv(args.output_dir / "equity_curves.csv", index_label="date")

    report = "\n".join(
        [
            "# Close-to-next-open backtest",
            "",
            f"Period requested: {args.start} through {args.end} (Yahoo end date is exclusive).",
            "Prices: Yahoo Finance adjusted daily OHLC, so splits and cash distributions are reflected.",
            "Execution: buy at each adjusted close, sell at the next adjusted open; fully in cash intraday.",
            "Costs: the stated basis points are charged at both the close buy and next-open sell.",
            "No taxes, commissions beyond modeled costs, capacity limits, or auction-fill constraints.",
            "",
            markdown_table(summary),
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
