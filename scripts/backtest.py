"""Run a historical backtest using the live bot's decision path."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import backtest
from src.backtest.simulator import SimulationResult, write_simulation_report
from src.config import ROOT, load_config
from src.data.market_data import get_history_many
from src.data.universe import load_universe

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-window-days", type=int, default=None)
    parser.add_argument("--test-window-days", type=int, default=None)
    parser.add_argument("--no-walk-forward", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(require_secrets=False)
    symbols = load_universe(cfg)
    if args.max_symbols is not None:
        symbols = symbols[: args.max_symbols]

    sim_days = int(args.years * 365.25)
    warmup_days = int(cfg.get("backtest", "warmup_days", default=280))
    start_date = pd.Timestamp(datetime.now(UTC).date() - timedelta(days=sim_days))

    log.info("fetching %d calendar days of history for %d symbols", sim_days + warmup_days, len(symbols))
    hist = get_history_many(cfg, symbols, days=sim_days + warmup_days)
    missing = [s for s in symbols if s not in hist]
    if missing:
        log.warning("missing history for %d symbols: %s", len(missing), ", ".join(missing[:20]))

    log.info("running live-path backtest on %d symbols", len(hist))
    result = backtest(
        cfg,
        hist,
        start_date=start_date,
        start_capital=args.start_capital,
        cost_bps=args.cost_bps,
        walk_forward=not args.no_walk_forward,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
    )

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "reports" / "backtests" / f"live_path_{stamp}"
    if result.summary is not None and result.trades_log is not None:
        write_simulation_report(
            SimulationResult(
                equity_curve=result.equity_curve.rename("equity").reset_index().rename(columns={"index": "date"}),
                trades=result.trades_log,
                summary=result.summary,
            ),
            out_dir,
        )

    print()
    if result.summary:
        print(f"  Period       : {result.summary['start_date']} to {result.summary['end_date']}")
    print(f"  Total return : {result.total_return * 100:>+8.2f}%")
    print(f"  CAGR         : {result.cagr * 100:>+8.2f}%")
    print(f"  Sharpe       : {result.sharpe:>+8.2f}")
    print(f"  Max drawdown : {result.max_drawdown * 100:>+8.2f}%")
    if result.summary:
        print(f"  Trades       : {result.trades} ({result.summary['buys']} buys, {result.summary['sells']} sells, {result.summary['stops']} stops)")
        print(f"  Win rate     : {result.summary['closed_win_rate'] * 100:>7.1f}%")
        if "walk_forward_windows" in result.summary:
            print(f"  WF windows   : {result.summary['walk_forward_windows']} train={result.summary['walk_forward_train_window_days']}d test={result.summary['walk_forward_test_window_days']}d")
    else:
        print(f"  Trades       : {result.trades}")
    print(f"  Final equity : ${result.equity_curve.iloc[-1]:,.0f}")
    print(f"  Report dir   : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
