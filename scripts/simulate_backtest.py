"""Run a five-year historical simulation using the live bot's current decisions."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.simulator import simulate_current_bot, write_simulation_report
from src.config import ROOT, load_config
from src.data.market_data import get_history_many
from src.data.universe import load_universe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(require_secrets=False)
    symbols = load_universe(cfg)
    if args.max_symbols is not None:
        symbols = symbols[: args.max_symbols]

    sim_days = int(args.years * 365.25)
    warmup_days = 280
    start_date = pd.Timestamp(datetime.now(UTC).date() - timedelta(days=sim_days))

    logging.info("fetching %d calendar days of history for %d symbols", sim_days + warmup_days, len(symbols))
    history = get_history_many(cfg, symbols, days=sim_days + warmup_days)
    missing = [s for s in symbols if s not in history]
    if missing:
        logging.warning("missing history for %d symbols: %s", len(missing), ", ".join(missing[:20]))

    result = simulate_current_bot(
        cfg,
        history,
        start_date=start_date,
        start_capital=args.start_capital,
        cost_bps=args.cost_bps,
    )

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "reports" / "backtests" / f"sim_{stamp}"
    write_simulation_report(result, out_dir)

    print()
    print(f"Period        : {result.summary['start_date']} to {result.summary['end_date']}")
    print(f"Symbols       : {result.summary['symbols']}")
    print(f"Total return  : {result.summary['total_return'] * 100:+.2f}%")
    print(f"CAGR          : {result.summary['cagr'] * 100:+.2f}%")
    print(f"Sharpe        : {result.summary['sharpe']:+.2f}")
    print(f"Max drawdown  : {result.summary['max_drawdown'] * 100:+.2f}%")
    print(f"Trades        : {result.summary['trades']} ({result.summary['buys']} buys, {result.summary['sells']} sells, {result.summary['stops']} stops)")
    print(f"Win rate      : {result.summary['closed_win_rate'] * 100:.1f}%")
    print(f"Final equity  : ${result.summary['final_equity']:,.0f}")
    print(f"Report dir    : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
