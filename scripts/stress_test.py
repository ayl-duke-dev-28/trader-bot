"""Run deterministic offline stress scenarios against the live-path simulator."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.stress import run_stress_suite, write_stress_report
from src.config import ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline market, data, transaction-cost, and macro stress scenarios."
    )
    parser.add_argument("--periods", type=int, default=520, help="synthetic business-day history")
    parser.add_argument("--simulation-bars", type=int, default=252, help="days in each measured scenario")
    parser.add_argument("--seed", type=int, default=7, help="deterministic random seed")
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--runtime-budget", type=float, default=10.0, help="warning budget per scenario in seconds")
    parser.add_argument("--no-model", action="store_true", help="disable the saved ML vote")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="run only this scenario; repeatable")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(require_secrets=False)
    suite = run_stress_suite(
        cfg,
        periods=args.periods,
        simulation_bars=args.simulation_bars,
        seed=args.seed,
        start_capital=args.start_capital,
        include_saved_model=not args.no_model,
        runtime_budget_seconds=args.runtime_budget,
        scenario_names=args.scenarios,
    )
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "reports" / "stress_tests" / stamp
    write_stress_report(suite, out_dir)

    print(f"Safety verdict : {suite.safety_verdict}")
    for result in suite.results:
        print(
            f"{result.scenario:18} {result.status:4} "
            f"return={result.total_return:+7.2%} drawdown={result.max_drawdown:7.2%} "
            f"worst_day={result.worst_day_return:7.2%} runtime={result.runtime_seconds:.2f}s"
        )
    print(f"Report dir     : {out_dir}")
    return 1 if suite.safety_verdict == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
