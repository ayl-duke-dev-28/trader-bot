"""Repeatedly score and tune the live-path bot on chronological data windows."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.optimizer import (
    AutoBacktestOptimizer,
    attach_holdout,
    combine_window_scores,
    optimizer_settings_from_config,
    parameter_specs_from_config,
    split_backtest_period,
    write_optimization_report,
)
from src.config import ROOT, Config, load_config

log = logging.getLogger(__name__)


def apply_optimized_config(optimized_path: Path, active_path: Path) -> Path:
    """Apply an explicitly selected config after preserving the current file."""
    optimized_path = Path(optimized_path)
    active_path = Path(active_path)
    if not optimized_path.is_file():
        raise FileNotFoundError(f"optimized config not found: {optimized_path}")
    if not active_path.is_file():
        raise FileNotFoundError(f"active config not found: {active_path}")
    parsed = yaml.safe_load(optimized_path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError("optimized config must contain a YAML mapping")

    backup = active_path.with_name(f"{active_path.name}.before-auto-backtest")
    if backup.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup = active_path.with_name(f"{active_path.name}.before-auto-backtest-{stamp}")
    shutil.copy2(active_path, backup)
    shutil.copy2(optimized_path, active_path)
    return backup


def _simulation_result(result: Any):
    from src.backtest.simulator import SimulationResult

    if result.summary is None or result.trades_log is None:
        raise ValueError("backtest did not return report details")
    curve = result.equity_diagnostics
    if curve is None:
        curve = result.equity_curve.rename("equity").reset_index().rename(columns={"index": "date"})
    return SimulationResult(curve, result.trades_log, result.summary)


def _macro_inputs(cfg: Config) -> tuple[pd.DataFrame | None, dict]:
    from src.data.macro import load_macro_panel, macro_cycle_history

    macro_cfg = dict(cfg.get("backtest", "macro_cycle", default={}) or {})
    if not bool(macro_cfg.get("enabled", False)):
        return None, macro_cfg
    configured_path = macro_cfg.get("data_path")
    if not configured_path:
        raise ValueError("backtest.macro_cycle is enabled but data_path is missing")
    path = Path(configured_path)
    if not path.is_absolute():
        path = ROOT / path
    panel = load_macro_panel(path)
    return macro_cycle_history(panel, macro_cfg), macro_cfg


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score, diagnose, tune, and re-backtest allow-listed bot settings; "
            "reserve the final period as an untouched holdout."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--years", type=float, default=8.0)
    parser.add_argument("--start-capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--target-score", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--min-improvement", type=float, default=None)
    parser.add_argument("--train-window-days", type=int, default=None)
    parser.add_argument("--test-window-days", type=int, default=None)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace the active config with the winner after saving a backup",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    from src.backtest.engine import BacktestResult, backtest
    from src.backtest.simulator import write_simulation_report
    from src.data.market_data import get_history_many
    from src.data.universe import load_universe

    if args.years <= 0:
        raise SystemExit("--years must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, require_secrets=False)
    settings = optimizer_settings_from_config(cfg)
    overrides = {
        "target_score": args.target_score,
        "max_iterations": args.max_iterations,
        "max_evaluations": args.max_evaluations,
        "min_improvement": args.min_improvement,
    }
    settings = replace(
        settings,
        **{name: value for name, value in overrides.items() if value is not None},
    )
    parameters = parameter_specs_from_config(cfg)

    auto_cfg = cfg.get("backtest", "auto_optimize", default={}) or {}
    development_fraction = float(auto_cfg.get("development_fraction", 0.60))
    validation_fraction = float(auto_cfg.get("validation_fraction", 0.20))
    end_date = pd.Timestamp(datetime.now(UTC).date())
    start_date = pd.Timestamp(
        datetime.now(UTC).date() - timedelta(days=int(args.years * 365.25))
    )
    windows = split_backtest_period(
        start_date,
        end_date,
        development_fraction=development_fraction,
        validation_fraction=validation_fraction,
    )

    symbols = load_universe(cfg)
    if args.max_symbols is not None:
        symbols = symbols[: args.max_symbols]
    configured_warmup = int(cfg.get("backtest", "warmup_days", default=280))
    train_days = args.train_window_days or int(
        cfg.get("backtest", "train_window_days", default=756)
    )
    warmup_days = configured_warmup if args.no_walk_forward else max(configured_warmup, train_days)
    history_days = int(args.years * 365.25) + warmup_days
    log.info("fetching %d calendar days for %d symbols once", history_days, len(symbols))
    history = get_history_many(cfg, symbols, days=history_days)
    if not history:
        raise RuntimeError("no market history was returned")
    missing = [symbol for symbol in symbols if symbol not in history]
    if missing:
        log.warning("missing history for %d symbols: %s", len(missing), ", ".join(missing[:20]))

    macro_cycles, macro_cfg = _macro_inputs(cfg)
    candidate_number = 0

    def run_window(candidate: Config, name: str) -> BacktestResult:
        window_start, window_end = windows[name]
        return backtest(
            candidate,
            history,
            start_date=window_start,
            end_date=window_end,
            start_capital=args.start_capital,
            cost_bps=args.cost_bps,
            walk_forward=not args.no_walk_forward,
            train_window_days=args.train_window_days,
            test_window_days=args.test_window_days,
            macro_cycles=macro_cycles,
            macro_cycle_config=macro_cfg,
        )

    def evaluate(candidate: Config):
        nonlocal candidate_number
        candidate_number += 1
        summaries = {}
        for name in ("development", "validation"):
            result = run_window(candidate, name)
            if result.summary is None:
                raise RuntimeError(f"{name} backtest returned no summary")
            summaries[name] = result.summary
        evaluation = combine_window_scores(summaries)
        print(
            f"Evaluation {candidate_number:>2}: {evaluation.assessment.score:>3}/100 "
            f"(development={evaluation.window_scores['development'].score}, "
            f"validation={evaluation.window_scores['validation'].score})"
        )
        for note in evaluation.assessment.weaknesses[:2]:
            print(f"  needs work: {note}")
        return evaluation

    optimizer = AutoBacktestOptimizer(cfg, evaluate, parameters, settings)
    optimized = optimizer.run()

    log.info("running the winning config once on the untouched holdout")
    holdout_result = run_window(optimized.best_config, "holdout")
    if holdout_result.summary is None:
        raise RuntimeError("holdout backtest returned no summary")
    optimized = attach_holdout(optimized, holdout_result.summary)

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "reports" / "backtests" / f"auto_optimize_{stamp}"
    write_optimization_report(optimized, out_dir)
    write_simulation_report(_simulation_result(holdout_result), out_dir / "holdout")

    print()
    print(f"Best tuning score : {optimized.best_score.score}/100")
    if optimized.holdout_score is not None:
        print(f"Holdout score     : {optimized.holdout_score.score}/100")
    print(f"Stop reason       : {optimized.stop_reason}")
    print(f"Candidate configs : {len(optimized.runs)}")
    print(f"Report directory  : {out_dir}")
    print("What worked:")
    for note in optimized.best_score.strengths:
        print(f"  + {note}")
    print("What needs work:")
    for note in optimized.best_score.weaknesses:
        print(f"  - {note}")

    if args.apply:
        backup = apply_optimized_config(out_dir / "optimized_config.yaml", args.config)
        print(f"Applied winner to : {args.config}")
        print(f"Previous config   : {backup}")
    else:
        print("Config not applied; inspect optimized_config.yaml or rerun with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
