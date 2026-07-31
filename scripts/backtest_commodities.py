"""Run a 10+ year, four-month walk-forward commodity ETF backtest."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.commodities import (
    benchmark_summary,
    default_commodity_candidates,
    run_commodity_walk_forward,
    write_commodity_report,
)
from src.config import ROOT, load_config
from src.data.market_data import get_history_many


COMMODITY_ETFS = {
    "GLD": "gold",
    "SLV": "silver",
    "USO": "US crude oil",
    "BNO": "Brent crude oil",
    "UNG": "natural gas",
    "DBA": "agriculture basket",
    "DBB": "industrial metals basket",
    "CPER": "copper",
    "PPLT": "platinum",
}
BENCHMARKS = ("DBC", "SPY")


def _load_prices_csv(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.apply(pd.to_numeric, errors="coerce").sort_index()


def _download_prices(years: float) -> pd.DataFrame:
    cfg = load_config(require_secrets=False)
    symbols = [*COMMODITY_ETFS, *BENCHMARKS]
    days = int(years * 365.25) + 45
    history = get_history_many(cfg, symbols, days=days)
    missing = [symbol for symbol in symbols if symbol not in history]
    if missing:
        raise RuntimeError(
            "missing historical prices for: " + ", ".join(missing) + ". "
            "Retry with network access or pass --prices-csv."
        )
    return pd.DataFrame({symbol: history[symbol]["close"] for symbol in symbols}).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=15.5, help="total history including training")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-months", type=int, default=4)
    parser.add_argument("--min-oos-years", type=float, default=10.0)
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--prices-csv", type=Path, default=None, help="wide adjusted-close CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "backtests" / "commodities_walk_forward_10y",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prices = _load_prices_csv(args.prices_csv) if args.prices_csv else _download_prices(args.years)
    missing_strategy = [symbol for symbol in COMMODITY_ETFS if symbol not in prices]
    if missing_strategy:
        raise RuntimeError("price input is missing commodity ETFs: " + ", ".join(missing_strategy))

    result = run_commodity_walk_forward(
        prices[list(COMMODITY_ETFS)],
        default_commodity_candidates(),
        train_years=args.train_years,
        test_months=args.test_months,
        cost_bps=args.cost_bps,
        start_capital=args.start_capital,
    )
    oos_years = (
        pd.Timestamp(result.summary["end_date"]) - pd.Timestamp(result.summary["start_date"])
    ).days / 365.25
    if oos_years < args.min_oos_years:
        raise RuntimeError(
            f"only {oos_years:.2f} out-of-sample years are available; "
            f"at least {args.min_oos_years:.2f} are required"
        )

    benchmarks = {
        symbol: benchmark_summary(
            prices[symbol],
            result.summary["start_date"],
            result.summary["end_date"],
            args.start_capital,
        )
        for symbol in BENCHMARKS
        if symbol in prices
    }
    write_commodity_report(result, args.out_dir, benchmarks)

    print(f"Period       : {result.summary['start_date']} to {result.summary['end_date']}")
    print(f"WF windows   : {result.summary['walk_forward_windows']} x {args.test_months} months")
    print(f"Total return : {result.summary['total_return']:+.2%}")
    print(f"CAGR         : {result.summary['cagr']:+.2%}")
    print(f"Sharpe       : {result.summary['sharpe']:+.2f}")
    print(f"Max drawdown : {result.summary['max_drawdown']:+.2%}")
    for symbol, values in benchmarks.items():
        print(
            f"{symbol:<12} : CAGR {values['cagr']:+.2%}, Sharpe {values['sharpe']:+.2f}, "
            f"max DD {values['max_drawdown']:+.2%}"
        )
    print(f"Report dir   : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
