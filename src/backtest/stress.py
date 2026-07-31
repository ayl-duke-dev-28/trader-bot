"""Deterministic, offline stress scenarios for the live-path simulator."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtest.simulator import SimulationResult, simulate_current_bot
from src.config import Config
from src.signals.ml import load_model


DEFAULT_SYMBOLS = ("QQQ", "AAPL", "NVDA", "MSFT", "AMD", "CRM", "PANW", "COIN")


@dataclass
class StressScenario:
    name: str
    description: str
    history: dict[str, pd.DataFrame]
    cost_bps: float = 5.0
    macro_cycles: pd.DataFrame | None = None
    macro_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class StressResult:
    scenario: str
    description: str
    status: str
    runtime_seconds: float
    final_equity: float
    total_return: float
    max_drawdown: float
    worst_day_return: float
    trades: int
    stops: int
    min_cash: float
    max_positions: int
    max_gross_exposure: float
    invariants_passed: bool
    failed_invariants: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StressSuiteResult:
    generated_at: str
    seed: int
    periods: int
    simulation_bars: int
    include_saved_model: bool
    total_runtime_seconds: float
    results: list[StressResult]

    def by_name(self, name: str) -> StressResult:
        for result in self.results:
            if result.scenario == name:
                return result
        raise KeyError(name)

    @property
    def safety_verdict(self) -> str:
        if any(result.status == "FAIL" for result in self.results):
            return "FAIL"
        if any(result.status == "WARN" for result in self.results):
            return "WARN"
        return "PASS"


def make_synthetic_history(
    *,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    periods: int = 520,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Create correlated OHLCV histories without network or broker access."""
    if periods < 220:
        raise ValueError("periods must be at least 220 for regime warmup")
    names = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    if "QQQ" not in names:
        raise ValueError("symbols must include QQQ")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2026-07-31", periods=periods, freq="B")
    market_returns = np.clip(rng.normal(0.0008, 0.008, periods), -0.08, 0.08)
    history: dict[str, pd.DataFrame] = {}

    for index, symbol in enumerate(names):
        if symbol == "QQQ":
            daily_returns = market_returns
            start_price = 300.0
        else:
            beta = 0.85 + 0.12 * (index % 5)
            idiosyncratic = rng.normal(0.00015 + index * 0.00002, 0.006 + index * 0.0004, periods)
            daily_returns = np.clip(beta * market_returns + idiosyncratic, -0.12, 0.12)
            start_price = 70.0 + index * 18.0

        close = start_price * np.cumprod(1.0 + daily_returns)
        prior_close = np.concatenate(([start_price], close[:-1]))
        overnight = rng.normal(0.0, 0.0025, periods)
        open_price = prior_close * (1.0 + overnight)
        intraday_range = rng.uniform(0.002, 0.015, periods)
        high = np.maximum(open_price, close) * (1.0 + intraday_range)
        low = np.minimum(open_price, close) * (1.0 - intraday_range)
        volume = rng.integers(750_000, 8_000_000, periods).astype(float)
        history[symbol] = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
    return history


def _copy_history(history: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {symbol: frame.copy(deep=True) for symbol, frame in history.items()}


def _apply_price_factor(
    history: dict[str, pd.DataFrame],
    factor: np.ndarray,
) -> dict[str, pd.DataFrame]:
    stressed = _copy_history(history)
    for frame in stressed.values():
        frame.loc[:, ["open", "high", "low", "close"]] = frame[
            ["open", "high", "low", "close"]
        ].mul(factor, axis=0)
    return stressed


def build_stress_scenarios(
    history: dict[str, pd.DataFrame],
    *,
    simulation_bars: int,
) -> dict[str, StressScenario]:
    """Build independent market, data, cost, and macro stress cases."""
    if not history or "QQQ" not in history:
        raise ValueError("stress history must include QQQ")
    periods = len(history["QQQ"])
    if simulation_bars < 20 or simulation_bars >= periods:
        raise ValueError("simulation_bars must be between 20 and periods - 1")

    baseline = _copy_history(history)

    flash_factor = np.ones(periods)
    flash_index = periods - max(2, simulation_bars // 2)
    flash_factor[flash_index:] = np.linspace(0.68, 0.80, periods - flash_index)
    flash = _apply_price_factor(history, flash_factor)

    bear_factor = np.ones(periods)
    bear_index = periods - simulation_bars + max(5, simulation_bars // 5)
    bear_factor[bear_index:] = np.linspace(1.0, 0.52, periods - bear_index)
    prolonged_bear = _apply_price_factor(history, bear_factor)

    volatile_factor = np.ones(periods)
    volatility_index = periods - simulation_bars + max(5, simulation_bars // 4)
    volatile_moves = np.resize(np.array([-0.055, 0.047, -0.035, 0.043]), periods - volatility_index)
    volatile_factor[volatility_index:] = np.exp(np.cumsum(volatile_moves))
    volatility_spike = _apply_price_factor(history, volatile_factor)

    missing_data = _copy_history(history)
    gap_symbols = [symbol for symbol in ("AAPL", "NVDA", "CRM") if symbol in missing_data]
    gap_start = periods - max(10, simulation_bars // 2)
    gap_rows = missing_data["QQQ"].index[gap_start::3]
    for symbol in gap_symbols:
        missing_data[symbol].loc[gap_rows, ["open", "high", "low", "close"]] = np.nan

    macro_start = history["QQQ"].index[-simulation_bars] - pd.Timedelta(days=1)
    macro_cycles = pd.DataFrame(
        {
            "long_score": [-0.75],
            "short_score": [-0.85],
            "composite_score": [-0.79],
            "regime": ["contraction"],
        },
        index=[macro_start],
    )
    macro_config = {
        "enabled": True,
        "neutral_max_gross_exposure": 0.60,
        "contraction_max_gross_exposure": 0.30,
    }

    return {
        "baseline": StressScenario(
            "baseline",
            "Correlated rising market with ordinary daily volatility and 5 bps costs.",
            baseline,
        ),
        "flash_crash": StressScenario(
            "flash_crash",
            "A synchronized 32% price gap followed by only a partial recovery.",
            flash,
        ),
        "prolonged_bear": StressScenario(
            "prolonged_bear",
            "A persistent decline that removes 48% from market prices.",
            prolonged_bear,
        ),
        "volatility_spike": StressScenario(
            "volatility_spike",
            "Alternating market-wide moves of roughly 3.5% to 5.5%.",
            volatility_spike,
        ),
        "missing_data": StressScenario(
            "missing_data",
            "Repeated OHLC gaps for several individual symbols.",
            missing_data,
        ),
        "high_cost": StressScenario(
            "high_cost",
            "Baseline prices with an extreme 100 bps cost on every trade.",
            _copy_history(history),
            cost_bps=100.0,
        ),
        "macro_contraction": StressScenario(
            "macro_contraction",
            "Severe long- and short-cycle contraction with a 30% gross cap.",
            _copy_history(history),
            macro_cycles=macro_cycles,
            macro_config=macro_config,
        ),
    }


def _stress_config(cfg: Config, *, enable_ml: bool) -> Config:
    raw = copy.deepcopy(cfg.raw)
    raw.setdefault("strategies", {}).setdefault("ml", {})["enabled"] = enable_ml
    return Config(raw=raw, api_key=cfg.api_key, api_secret=cfg.api_secret, is_live=False)


def _run_scenario(
    cfg: Config,
    scenario: StressScenario,
    *,
    start_date: pd.Timestamp,
    start_capital: float,
    model_bundle: dict[str, Any] | None,
    runtime_budget_seconds: float,
) -> StressResult:
    started = time.perf_counter()
    try:
        simulation = simulate_current_bot(
            cfg,
            scenario.history,
            start_date=start_date,
            start_capital=start_capital,
            cost_bps=scenario.cost_bps,
            model_bundle=model_bundle,
            macro_cycles=scenario.macro_cycles,
            macro_cycle_config=scenario.macro_config,
        )
        runtime = time.perf_counter() - started
        return _evaluate_simulation(
            cfg,
            scenario,
            simulation,
            runtime_seconds=runtime,
            runtime_budget_seconds=runtime_budget_seconds,
        )
    except Exception as exc:
        return StressResult(
            scenario=scenario.name,
            description=scenario.description,
            status="FAIL",
            runtime_seconds=time.perf_counter() - started,
            final_equity=float("nan"),
            total_return=float("nan"),
            max_drawdown=float("nan"),
            worst_day_return=float("nan"),
            trades=0,
            stops=0,
            min_cash=float("nan"),
            max_positions=0,
            max_gross_exposure=float("nan"),
            invariants_passed=False,
            failed_invariants=[f"simulation raised {type(exc).__name__}: {exc}"],
        )


def _evaluate_simulation(
    cfg: Config,
    scenario: StressScenario,
    simulation: SimulationResult,
    *,
    runtime_seconds: float,
    runtime_budget_seconds: float,
) -> StressResult:
    curve = simulation.equity_curve
    equity = pd.to_numeric(curve["equity"], errors="coerce")
    cash = pd.to_numeric(curve["cash"], errors="coerce")
    positions = pd.to_numeric(curve["positions"], errors="coerce")
    gross = ((equity - cash) / equity.replace(0, np.nan)).clip(lower=0.0)
    configured_max_positions = int(cfg.get("risk", "max_positions", default=20))
    configured_max_gross = float(cfg.get("risk", "max_gross_exposure", default=0.80))
    if scenario.macro_config.get("enabled"):
        configured_max_gross = min(
            configured_max_gross,
            float(scenario.macro_config["contraction_max_gross_exposure"]),
        )

    checks = {
        "equity is finite": bool(np.isfinite(equity).all()),
        "equity remains positive": bool((equity > 0).all()),
        "cash remains nonnegative": bool((cash >= -0.01).all()),
        "position count stays within configured cap": bool(
            (positions <= configured_max_positions).all()
        ),
        "gross exposure stays within active cap": bool(
            (gross <= configured_max_gross + 0.002).all()
        ),
    }
    failures = [label for label, passed in checks.items() if not passed]
    warnings: list[str] = []
    max_drawdown = float(simulation.summary["max_drawdown"])
    worst_day = float(simulation.summary["worst_day_return"])
    if max_drawdown < -0.35:
        warnings.append(f"max drawdown {max_drawdown:.1%} exceeds the 35% stress budget")
    if worst_day < -0.20:
        warnings.append(f"worst day {worst_day:.1%} exceeds the 20% stress budget")
    if runtime_seconds > runtime_budget_seconds:
        warnings.append(
            f"runtime {runtime_seconds:.2f}s exceeds the {runtime_budget_seconds:.2f}s scenario budget"
        )

    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return StressResult(
        scenario=scenario.name,
        description=scenario.description,
        status=status,
        runtime_seconds=runtime_seconds,
        final_equity=float(simulation.summary["final_equity"]),
        total_return=float(simulation.summary["total_return"]),
        max_drawdown=max_drawdown,
        worst_day_return=worst_day,
        trades=int(simulation.summary["trades"]),
        stops=int(simulation.summary["stops"]),
        min_cash=float(cash.min()),
        max_positions=int(positions.max()),
        max_gross_exposure=float(gross.max()),
        invariants_passed=not failures,
        failed_invariants=failures,
        warnings=warnings,
    )


def run_stress_suite(
    cfg: Config,
    *,
    periods: int = 520,
    simulation_bars: int = 252,
    seed: int = 7,
    start_capital: float = 100_000.0,
    include_saved_model: bool = True,
    runtime_budget_seconds: float = 10.0,
    scenario_names: Iterable[str] | None = None,
) -> StressSuiteResult:
    """Run deterministic live-path simulations and collect safety/risk metrics."""
    started = time.perf_counter()
    history = make_synthetic_history(periods=periods, seed=seed)
    scenarios = build_stress_scenarios(history, simulation_bars=simulation_bars)
    selected_names = tuple(scenario_names or scenarios.keys())
    unknown = [name for name in selected_names if name not in scenarios]
    if unknown:
        raise ValueError(f"unknown stress scenarios: {', '.join(unknown)}")

    model_bundle = load_model(cfg) if include_saved_model else None
    enable_ml = bool(model_bundle is not None and include_saved_model)
    simulation_cfg = _stress_config(cfg, enable_ml=enable_ml)
    start_date = history["QQQ"].index[-simulation_bars]
    results = [
        _run_scenario(
            simulation_cfg,
            scenarios[name],
            start_date=start_date,
            start_capital=start_capital,
            model_bundle=model_bundle,
            runtime_budget_seconds=runtime_budget_seconds,
        )
        for name in selected_names
    ]
    return StressSuiteResult(
        generated_at=datetime.now(UTC).isoformat(),
        seed=seed,
        periods=periods,
        simulation_bars=simulation_bars,
        include_saved_model=enable_ml,
        total_runtime_seconds=time.perf_counter() - started,
        results=results,
    )


def write_stress_report(suite: StressSuiteResult, out_dir: Path) -> None:
    """Write a concise Markdown report plus CSV and JSON machine output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in suite.results]
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    payload = {
        "generated_at": suite.generated_at,
        "seed": suite.seed,
        "periods": suite.periods,
        "simulation_bars": suite.simulation_bars,
        "include_saved_model": suite.include_saved_model,
        "total_runtime_seconds": suite.total_runtime_seconds,
        "safety_verdict": suite.safety_verdict,
        "results": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Trader bot stress report",
        "",
        f"- Generated: `{suite.generated_at}`",
        f"- Seed: `{suite.seed}`",
        f"- Synthetic history: `{suite.periods}` business days; simulated `{suite.simulation_bars}` days",
        f"- Saved ML model included: `{'yes' if suite.include_saved_model else 'no'}`",
        f"- Total runtime: `{suite.total_runtime_seconds:.2f}s`",
        f"- Safety verdict: **{suite.safety_verdict}**",
        "",
        "| Scenario | Status | Return | Max drawdown | Worst day | Trades | Stops | Max gross | Runtime |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in suite.results:
        lines.append(
            f"| {result.scenario} | {result.status} | {result.total_return:.2%} | "
            f"{result.max_drawdown:.2%} | {result.worst_day_return:.2%} | "
            f"{result.trades} | {result.stops} | {result.max_gross_exposure:.2%} | "
            f"{result.runtime_seconds:.2f}s |"
        )
    details = [result for result in suite.results if result.failed_invariants or result.warnings]
    if details:
        lines.extend(["", "## Findings", ""])
        for result in details:
            for finding in [*result.failed_invariants, *result.warnings]:
                lines.append(f"- **{result.scenario}:** {finding}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PASS means all software safety invariants and configured exposure limits held. "
            "WARN means the simulation stayed operational but crossed a stated risk or runtime budget. "
            "FAIL means an exception, insolvency, non-finite accounting value, negative cash, or cap violation.",
            "",
            "Synthetic scenarios test behavior under controlled shocks; they do not predict returns or replace "
            "historical out-of-sample backtests.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
