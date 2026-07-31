"""Run a 10+ year, four-month walk-forward commodity ETF backtest."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.commodities import (
    benchmark_summary,
    default_commodity_candidates,
    default_diversified_candidates,
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
BENCHMARKS = ("DBC", "SPY", "VOO", "QQQ", "QQQM")


def _load_prices_csv(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return prices.apply(pd.to_numeric, errors="coerce").sort_index()


def _download_prices(years: float) -> pd.DataFrame:
    cfg = load_config(require_secrets=False)
    symbols = [*COMMODITY_ETFS, *BENCHMARKS]
    days = int(years * 365.25) + 45
    history = get_history_many(cfg, symbols, days=days, allow_partial_cache=True)
    missing_strategy = [symbol for symbol in COMMODITY_ETFS if symbol not in history]
    if missing_strategy:
        raise RuntimeError(
            "missing historical prices for: " + ", ".join(missing_strategy) + ". "
            "Retry with network access or pass --prices-csv."
        )
    missing_benchmarks = [symbol for symbol in BENCHMARKS if symbol not in history]
    if missing_benchmarks:
        logging.warning("missing optional benchmarks: %s", ", ".join(missing_benchmarks))
    return pd.DataFrame({symbol: frame["close"] for symbol, frame in history.items()}).sort_index()


def _write_comparison(out_dir: Path, results, benchmarks) -> None:
    baseline = results["baseline"].summary
    diversified = results["diversified"].summary
    payload = {
        "baseline": baseline,
        "diversified": diversified,
        "benchmarks": benchmarks,
        "diversified_minus_baseline": {
            key: float(diversified[key]) - float(baseline[key])
            for key in ("total_return", "cagr", "sharpe", "max_drawdown", "annualized_volatility")
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Commodity strategy comparison",
        "",
        "| Metric | Baseline | Diversified | Difference |",
        "| --- | ---: | ---: | ---: |",
        f"| Total return | {baseline['total_return']:.2%} | {diversified['total_return']:.2%} | {diversified['total_return'] - baseline['total_return']:+.2%} |",
        f"| CAGR | {baseline['cagr']:.2%} | {diversified['cagr']:.2%} | {diversified['cagr'] - baseline['cagr']:+.2%} |",
        f"| Sharpe | {baseline['sharpe']:.2f} | {diversified['sharpe']:.2f} | {diversified['sharpe'] - baseline['sharpe']:+.2f} |",
        f"| Max drawdown | {baseline['max_drawdown']:.2%} | {diversified['max_drawdown']:.2%} | {diversified['max_drawdown'] - baseline['max_drawdown']:+.2%} |",
        f"| Volatility | {baseline['annualized_volatility']:.2%} | {diversified['annualized_volatility']:.2%} | {diversified['annualized_volatility'] - baseline['annualized_volatility']:+.2%} |",
        f"| Average exposure | {baseline['average_gross_exposure']:.2%} | {diversified['average_gross_exposure']:.2%} | {diversified['average_gross_exposure'] - baseline['average_gross_exposure']:+.2%} |",
        f"| Transaction costs | ${baseline['transaction_costs']:,.0f} | ${diversified['transaction_costs']:,.0f} | ${diversified['transaction_costs'] - baseline['transaction_costs']:+,.0f} |",
        "",
        "The diversified variant selects five to nine qualifying funds, caps each fund at 25%, and caps every commodity group at 35%.",
    ]
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=15.5, help="total history including training")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-months", type=int, default=4)
    parser.add_argument("--min-oos-years", type=float, default=10.0)
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--variant",
        choices=("baseline", "diversified", "compare"),
        default="compare",
        help="run the original strategy, the diversified variant, or both",
    )
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

    candidate_sets = {
        "baseline": default_commodity_candidates(),
        "diversified": default_diversified_candidates(),
    }
    variants = ("baseline", "diversified") if args.variant == "compare" else (args.variant,)
    results = {}
    for variant in variants:
        result = run_commodity_walk_forward(
            prices[list(COMMODITY_ETFS)],
            candidate_sets[variant],
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
        results[variant] = result

    reference = next(iter(results.values()))
    benchmarks = {
        symbol: benchmark_summary(
            prices[symbol],
            reference.summary["start_date"],
            reference.summary["end_date"],
            args.start_capital,
        )
        for symbol in BENCHMARKS
        if symbol in prices
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(args.out_dir / "adjusted_closes.csv", index_label="date")
    for variant, result in results.items():
        variant_dir = args.out_dir / variant if args.variant == "compare" else args.out_dir
        write_commodity_report(result, variant_dir, benchmarks)
        print(f"\n{variant.upper()}")
        print(f"Period       : {result.summary['start_date']} to {result.summary['end_date']}")
        print(f"WF windows   : {result.summary['walk_forward_windows']} x {args.test_months} months")
        print(f"Total return : {result.summary['total_return']:+.2%}")
        print(f"CAGR         : {result.summary['cagr']:+.2%}")
        print(f"Sharpe       : {result.summary['sharpe']:+.2f}")
        print(f"Max drawdown : {result.summary['max_drawdown']:+.2%}")
    if args.variant == "compare":
        _write_comparison(args.out_dir, results, benchmarks)

    print("\nBENCHMARKS")
    for symbol, values in benchmarks.items():
        print(
            f"{symbol:<12} : {values['start_date']} to {values['end_date']}, "
            f"total {values['total_return']:+.2%}, CAGR {values['cagr']:+.2%}, Sharpe {values['sharpe']:+.2f}, "
            f"max DD {values['max_drawdown']:+.2%}"
        )
    print(f"Report dir   : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
