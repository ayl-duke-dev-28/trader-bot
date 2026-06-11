"""Run a walk-forward backtest on the configured universe."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import backtest
from src.config import load_config
from src.data.market_data import get_history_many
from src.data.universe import load_universe

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    symbols = load_universe(cfg)
    log.info("fetching history for %d symbols...", len(symbols))
    hist = get_history_many(cfg, symbols)
    log.info("running backtest on %d symbols", len(hist))
    result = backtest(cfg, hist)
    print()
    print(f"  Total return : {result.total_return * 100:>+8.2f}%")
    print(f"  CAGR         : {result.cagr * 100:>+8.2f}%")
    print(f"  Sharpe       : {result.sharpe:>+8.2f}")
    print(f"  Max drawdown : {result.max_drawdown * 100:>+8.2f}%")
    print(f"  Rebalances   : {result.trades}")
    print(f"  Final equity : ${result.equity_curve.iloc[-1]:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
