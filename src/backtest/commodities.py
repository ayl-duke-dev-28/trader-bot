"""Commodity ETF strategy and leakage-safe walk-forward backtesting.

The strategy is deliberately separate from the live tech-equity path. It uses
absolute and cross-sectional momentum, inverse-volatility sizing, a portfolio
volatility target, monthly rebalancing, and cash when no commodity is trending
up. Walk-forward parameter selection sees only the rolling training window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Callable, Iterable
import json

import numpy as np
import pandas as pd


COMMODITY_GROUPS = {
    "GLD": "precious_metals",
    "SLV": "precious_metals",
    "PPLT": "precious_metals",
    "USO": "energy",
    "BNO": "energy",
    "UNG": "energy",
    "DBA": "agriculture",
    "DBB": "industrial_metals",
    "CPER": "industrial_metals",
}


@dataclass(frozen=True)
class CommodityParameters:
    momentum_days: int
    top_n: int
    trend_days: int = 200
    volatility_days: int = 60
    target_volatility: float = 0.15
    max_position: float = 0.40
    max_group: float = 1.00


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class CommodityBacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    windows: pd.DataFrame
    summary: dict[str, float | int | str]


ParameterSelector = Callable[
    [pd.DataFrame, list[CommodityParameters], float],
    tuple[CommodityParameters, dict[str, float]],
]


def build_walk_forward_windows(
    index: pd.Index,
    train_years: int = 5,
    test_months: int = 4,
) -> list[WalkForwardWindow]:
    """Build rolling calendar windows with training strictly before testing."""
    dates = pd.DatetimeIndex(pd.to_datetime(index)).dropna().sort_values().unique()
    if dates.empty:
        raise ValueError("price index is empty")
    if train_years <= 0 or test_months <= 0:
        raise ValueError("train_years and test_months must be positive")

    first_test_start = pd.Timestamp(dates.min()) + pd.DateOffset(years=train_years)
    last_date = pd.Timestamp(dates.max())
    windows: list[WalkForwardWindow] = []
    test_start = first_test_start
    while test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1) <= last_date:
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        windows.append(
            WalkForwardWindow(
                train_start=test_start - pd.DateOffset(years=train_years),
                train_end=test_start - pd.Timedelta(days=1),
                test_start=test_start,
                test_end=test_end,
            )
        )
        test_start = test_start + pd.DateOffset(months=test_months)
    return windows


def _capped_weights(raw: pd.Series, cap: float) -> pd.Series:
    """Normalize positive scores while respecting a per-position cap."""
    raw = raw.clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0 or cap <= 0:
        return pd.Series(0.0, index=raw.index)
    weights = raw / raw.sum()
    result = pd.Series(0.0, index=raw.index)
    remaining = 1.0
    active = list(raw[raw > 0].index)
    while active and remaining > 1e-12:
        active_raw = raw.loc[active]
        allocation = remaining * active_raw / active_raw.sum()
        capped = allocation[allocation > cap + 1e-12]
        if capped.empty:
            result.loc[active] = allocation
            break
        for symbol in capped.index:
            result.loc[symbol] = cap
            remaining -= cap
            active.remove(symbol)
    return result


def _apply_group_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Reduce correlated commodity groups to a shared cap, leaving excess cash."""
    if cap >= 1.0:
        return weights
    result = weights.copy()
    groups: dict[str, list[str]] = {}
    for symbol in result.index:
        group = COMMODITY_GROUPS.get(str(symbol).upper(), str(symbol).upper())
        groups.setdefault(group, []).append(symbol)
    for symbols in groups.values():
        group_weight = float(result.loc[symbols].sum())
        if group_weight > cap:
            result.loc[symbols] *= cap / group_weight
    return result


def commodity_target_weights(
    prices: pd.DataFrame,
    parameters: CommodityParameters,
) -> pd.Series:
    """Return next-session long-only ETF weights using data through this close."""
    if prices.empty:
        return pd.Series(0.0, index=prices.columns, dtype=float)
    clean = prices.astype(float).replace([np.inf, -np.inf], np.nan)
    output = pd.Series(0.0, index=clean.columns, dtype=float)
    required = max(parameters.momentum_days + 1, parameters.trend_days, parameters.volatility_days + 1)

    eligible: dict[str, tuple[float, float]] = {}
    for symbol in clean.columns:
        close = clean[symbol].dropna()
        if len(close) < required or close.iloc[-1] <= 0:
            continue
        momentum = close.iloc[-1] / close.iloc[-parameters.momentum_days - 1] - 1.0
        trend = close.iloc[-parameters.trend_days:].mean()
        log_returns = np.log(close / close.shift(1)).dropna().iloc[-parameters.volatility_days:]
        annual_vol = float(log_returns.std(ddof=1) * sqrt(252))
        if momentum > 0 and close.iloc[-1] > trend and np.isfinite(annual_vol) and annual_vol > 0:
            eligible[symbol] = (float(momentum), annual_vol)

    if not eligible:
        return output
    selected = sorted(eligible, key=lambda symbol: eligible[symbol][0], reverse=True)[: parameters.top_n]
    inverse_vol = pd.Series({symbol: 1.0 / eligible[symbol][1] for symbol in selected})
    weights = _capped_weights(inverse_vol, parameters.max_position)
    weights = _apply_group_cap(weights, parameters.max_group)

    recent_returns = clean[selected].pct_change(fill_method=None).dropna(how="any").iloc[-parameters.volatility_days:]
    if len(recent_returns) >= 2:
        covariance = recent_returns.cov() * 252
        vector = weights.reindex(selected).fillna(0.0).to_numpy()
        variance = float(vector @ covariance.to_numpy() @ vector)
        estimated_vol = sqrt(max(0.0, variance))
        if estimated_vol > 0:
            weights *= min(1.0, parameters.target_volatility / estimated_vol)

    output.loc[weights.index] = weights
    return output.clip(lower=0.0)


def _metrics(
    equity: pd.Series,
    daily_returns: pd.Series,
    *,
    initial_equity: float | None = None,
) -> dict[str, float]:
    clean_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        raise ValueError("equity curve is empty")
    base_equity = float(equity.iloc[0] if initial_equity is None else initial_equity)
    total_return = float(equity.iloc[-1] / base_equity - 1.0)
    elapsed_days = max(1, (equity.index[-1] - equity.index[0]).days)
    years = elapsed_days / 365.25
    cagr = float((equity.iloc[-1] / base_equity) ** (1.0 / years) - 1.0)
    volatility = float(clean_returns.std(ddof=1) * sqrt(252)) if len(clean_returns) > 1 else 0.0
    sharpe = (
        float(clean_returns.mean() / clean_returns.std(ddof=1) * sqrt(252))
        if len(clean_returns) > 1 and clean_returns.std(ddof=1) > 0
        else 0.0
    )
    drawdown_curve = pd.concat(
        [pd.Series([base_equity], index=[equity.index[0] - pd.Timedelta(microseconds=1)]), equity]
    )
    max_drawdown = float((drawdown_curve / drawdown_curve.cummax() - 1.0).min())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _simulate_parameter(
    prices: pd.DataFrame,
    parameters: CommodityParameters,
    cost_bps: float,
) -> dict[str, float]:
    """Run an in-sample candidate simulation for parameter selection."""
    returns = prices.pct_change(fill_method=None)
    weights = pd.Series(0.0, index=prices.columns)
    equity_values: list[float] = []
    daily_values: list[float] = []
    equity = 1.0
    previous_month: tuple[int, int] | None = None
    for date in prices.index:
        day_return = float((weights * returns.loc[date].fillna(0.0)).sum())
        equity *= max(0.0, 1.0 + day_return)
        month = (date.year, date.month)
        if month != previous_month:
            target = commodity_target_weights(prices.loc[:date], parameters)
            turnover = float((target - weights).abs().sum())
            equity *= max(0.0, 1.0 - turnover * cost_bps / 10_000.0)
            weights = target
            previous_month = month
        equity_values.append(equity)
        daily_values.append(day_return)
    curve = pd.Series(equity_values, index=prices.index)
    metrics = _metrics(curve, pd.Series(daily_values, index=prices.index), initial_equity=1.0)
    metrics["final_equity"] = float(curve.iloc[-1])
    return metrics


def select_commodity_parameters(
    train_prices: pd.DataFrame,
    candidates: list[CommodityParameters],
    cost_bps: float,
) -> tuple[CommodityParameters, dict[str, float]]:
    """Choose the prior-window candidate with the best cost-adjusted Sharpe."""
    if not candidates:
        raise ValueError("at least one parameter candidate is required")
    scored: list[tuple[float, float, float, CommodityParameters, dict[str, float]]] = []
    for candidate in candidates:
        metrics = _simulate_parameter(train_prices, candidate, cost_bps)
        scored.append(
            (
                metrics["sharpe"],
                metrics["cagr"],
                metrics["max_drawdown"],
                candidate,
                metrics,
            )
        )
    _, _, _, selected, metrics = max(scored, key=lambda row: (row[0], row[1], row[2]))
    return selected, {f"train_{key}": value for key, value in metrics.items()}


def default_commodity_candidates() -> list[CommodityParameters]:
    return [
        CommodityParameters(momentum_days=momentum, top_n=top_n)
        for momentum in (63, 126, 252)
        for top_n in (2, 3, 4)
    ]


def default_diversified_candidates() -> list[CommodityParameters]:
    """Broader variants with smaller single-name and commodity-group limits."""
    return [
        CommodityParameters(
            momentum_days=momentum,
            top_n=top_n,
            max_position=0.25,
            max_group=0.35,
        )
        for momentum in (63, 126, 252)
        for top_n in (5, 7, 9)
    ]


def run_commodity_walk_forward(
    prices: pd.DataFrame,
    candidates: Iterable[CommodityParameters] | None = None,
    *,
    train_years: int = 5,
    test_months: int = 4,
    cost_bps: float = 10.0,
    start_capital: float = 100_000.0,
    parameter_selector: ParameterSelector = select_commodity_parameters,
) -> CommodityBacktestResult:
    """Select on rolling history and trade each following four-month window."""
    if start_capital <= 0 or cost_bps < 0:
        raise ValueError("start_capital must be positive and cost_bps non-negative")
    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]
    clean = clean.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if clean.empty or len(clean.columns) == 0:
        raise ValueError("prices are empty")

    candidate_list = list(candidates or default_commodity_candidates())
    windows = build_walk_forward_windows(clean.index, train_years, test_months)
    selected_by_date: list[tuple[WalkForwardWindow, CommodityParameters]] = []
    window_rows: list[dict[str, float | int | str]] = []
    for window_number, window in enumerate(windows, start=1):
        train = clean.loc[window.train_start : window.train_end].dropna(how="all")
        test = clean.loc[window.test_start : window.test_end].dropna(how="all")
        if train.empty or test.empty:
            continue
        train.attrs["test_start"] = window.test_start
        selected, train_metrics = parameter_selector(train, candidate_list, cost_bps)
        selected_by_date.append((window, selected))
        window_rows.append(
            {
                "window": window_number,
                "train_start": window.train_start.date().isoformat(),
                "train_end": window.train_end.date().isoformat(),
                "test_start": window.test_start.date().isoformat(),
                "test_end": window.test_end.date().isoformat(),
                **asdict(selected),
                **train_metrics,
            }
        )
    if not selected_by_date:
        raise ValueError("not enough price history to create a walk-forward test window")

    first_test = selected_by_date[0][0].test_start
    last_test = selected_by_date[-1][0].test_end
    test_prices = clean.loc[first_test:last_test]
    asset_returns = clean.pct_change(fill_method=None).reindex(test_prices.index)
    weights = pd.Series(0.0, index=clean.columns)
    equity = start_capital
    previous_month: tuple[int, int] | None = None
    active_window: WalkForwardWindow | None = None
    equity_rows: list[tuple[pd.Timestamp, float]] = []
    daily_rows: list[tuple[pd.Timestamp, float]] = []
    weight_rows: list[pd.Series] = []
    total_turnover = 0.0
    transaction_costs = 0.0
    rebalances = 0
    equity_before_date: dict[pd.Timestamp, float] = {}

    for date in test_prices.index:
        match = next(
            ((window, params) for window, params in selected_by_date if window.test_start <= date <= window.test_end),
            None,
        )
        if match is None:
            continue
        window, parameters = match
        equity_before_date[date] = equity
        day_return = float((weights * asset_returns.loc[date].fillna(0.0)).sum())
        equity *= max(0.0, 1.0 + day_return)
        month = (date.year, date.month)
        window_changed = active_window != window
        if window_changed or month != previous_month:
            target = commodity_target_weights(clean.loc[:date], parameters)
            turnover = float((target - weights).abs().sum())
            cost = equity * turnover * cost_bps / 10_000.0
            equity = max(0.0, equity - cost)
            transaction_costs += cost
            total_turnover += turnover
            rebalances += 1
            weights = target
            previous_month = month
            active_window = window
        equity_rows.append((date, equity))
        daily_rows.append((date, day_return))
        row = weights.copy()
        row.name = date
        weight_rows.append(row)

    equity_curve = pd.Series(dict(equity_rows), name="equity", dtype=float)
    daily_returns = equity_curve.pct_change(fill_method=None)
    if not daily_returns.empty:
        daily_returns.iloc[0] = equity_curve.iloc[0] / start_capital - 1.0
    daily_returns.name = "return"
    weights_frame = pd.DataFrame(weight_rows).reindex(columns=clean.columns).fillna(0.0)
    windows_frame = pd.DataFrame(window_rows)
    for row_index, row in windows_frame.iterrows():
        window_start = pd.Timestamp(row["test_start"])
        window_end = pd.Timestamp(row["test_end"])
        window_curve = equity_curve.loc[window_start:window_end]
        if window_curve.empty:
            windows_frame.loc[row_index, "test_observations"] = 0
            windows_frame.loc[row_index, "test_return"] = np.nan
            windows_frame.loc[row_index, "test_sharpe"] = np.nan
            windows_frame.loc[row_index, "test_max_drawdown"] = np.nan
            continue
        initial = equity_before_date[window_curve.index[0]]
        window_returns = daily_returns.loc[window_curve.index]
        window_metrics = _metrics(window_curve, window_returns, initial_equity=initial)
        windows_frame.loc[row_index, "test_observations"] = len(window_curve)
        windows_frame.loc[row_index, "test_return"] = window_metrics["total_return"]
        windows_frame.loc[row_index, "test_sharpe"] = window_metrics["sharpe"]
        windows_frame.loc[row_index, "test_max_drawdown"] = window_metrics["max_drawdown"]
    summary: dict[str, float | int | str] = {
        **_metrics(equity_curve, daily_returns, initial_equity=start_capital),
        "start_date": equity_curve.index.min().date().isoformat(),
        "end_date": equity_curve.index.max().date().isoformat(),
        "start_capital": float(start_capital),
        "final_equity": float(equity_curve.iloc[-1]),
        "walk_forward_windows": len(window_rows),
        "train_years": train_years,
        "test_months": test_months,
        "cost_bps": float(cost_bps),
        "rebalances": rebalances,
        "turnover": total_turnover,
        "transaction_costs": transaction_costs,
        "average_gross_exposure": float(weights_frame.sum(axis=1).mean()),
    }
    return CommodityBacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights=weights_frame,
        windows=windows_frame,
        summary=summary,
    )


def benchmark_summary(
    prices: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    start_capital: float = 100_000.0,
) -> dict[str, float | str]:
    series = prices.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].dropna()
    if series.empty:
        raise ValueError("benchmark has no prices in the test period")
    returns = series.pct_change(fill_method=None).fillna(0.0)
    equity = start_capital * (1.0 + returns).cumprod()
    return {
        **_metrics(equity, returns, initial_equity=start_capital),
        "start_date": series.index.min().date().isoformat(),
        "end_date": series.index.max().date().isoformat(),
        "final_equity": float(equity.iloc[-1]),
    }


def write_commodity_report(
    result: CommodityBacktestResult,
    out_dir: Path,
    benchmarks: dict[str, dict[str, float | str]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out_dir / "equity_curve.csv", header=True)
    result.daily_returns.to_csv(out_dir / "daily_returns.csv", header=True)
    result.weights.to_csv(out_dir / "weights.csv", index_label="date")
    result.windows.to_csv(out_dir / "walk_forward_windows.csv", index=False)
    payload = {"strategy": result.summary, "benchmarks": benchmarks or {}}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Commodity walk-forward backtest",
        "",
        f"- Period: {result.summary['start_date']} to {result.summary['end_date']}",
        f"- Four-month out-of-sample windows: {result.summary['walk_forward_windows']}",
        f"- Final equity: ${result.summary['final_equity']:,.2f}",
        f"- Total return: {result.summary['total_return']:.2%}",
        f"- CAGR: {result.summary['cagr']:.2%}",
        f"- Sharpe: {result.summary['sharpe']:.2f}",
        f"- Max drawdown: {result.summary['max_drawdown']:.2%}",
        f"- Average gross exposure: {result.summary['average_gross_exposure']:.2%}",
        f"- Transaction costs: ${result.summary['transaction_costs']:,.2f}",
    ]
    if benchmarks:
        lines.extend(["", "## Benchmarks", ""])
        for name, values in benchmarks.items():
            lines.append(
                f"- {name}: CAGR {values['cagr']:.2%}, Sharpe {values['sharpe']:.2f}, "
                f"max drawdown {values['max_drawdown']:.2%}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
