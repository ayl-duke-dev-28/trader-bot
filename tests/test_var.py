"""Portfolio historical VaR and expected-shortfall tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.simulator import simulate_current_bot
from src.broker.alpaca_client import Account
from src.risk.manager import RiskManager, TradeIntent
from src.risk.var import estimate_historical_risk, filter_buy_intents_by_var


def _history_from_returns(returns: list[float], symbol: str = "RISK") -> dict[str, pd.DataFrame]:
    prices = [100.0]
    for daily_return in returns:
        prices.append(prices[-1] * (1.0 + daily_return))
    index = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    close = np.asarray(prices)
    return {
        symbol: pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=index,
        )
    }


def _var_config(**overrides):
    config = {
        "enabled": True,
        "confidence": 0.80,
        "lookback_days": 100,
        "min_observations": 5,
        "max_var_pct": 0.02,
        "max_expected_shortfall_pct": 0.03,
        "fail_closed": True,
    }
    config.update(overrides)
    return config


def test_historical_var_and_expected_shortfall_match_portfolio_loss_distribution():
    returns = [-0.10, -0.04, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    estimate = estimate_historical_risk(
        _history_from_returns(returns),
        exposures={"RISK": 50_000.0},
        equity=100_000.0,
        confidence=0.80,
        lookback_days=100,
        min_observations=5,
    )

    portfolio_losses = -0.5 * np.asarray(returns)
    expected_var = float(np.quantile(portfolio_losses, 0.80))
    expected_es = float(portfolio_losses[portfolio_losses >= expected_var].mean())
    assert estimate is not None
    assert estimate.observations == len(returns)
    assert np.isclose(estimate.var_pct, expected_var)
    assert np.isclose(estimate.expected_shortfall_pct, expected_es)


def test_portfolio_var_preserves_cross_symbol_diversification():
    left = _history_from_returns([-0.04, 0.04] * 20, "LEFT")
    right = _history_from_returns([0.04, -0.04] * 20, "RIGHT")
    estimate = estimate_historical_risk(
        {**left, **right},
        exposures={"LEFT": 50_000.0, "RIGHT": 50_000.0},
        equity=100_000.0,
        confidence=0.95,
        lookback_days=40,
        min_observations=20,
    )

    assert estimate is not None
    assert estimate.var_pct < 0.001
    assert estimate.expected_shortfall_pct < 0.001


def test_var_returns_none_for_invalid_equity_or_insufficient_aligned_history():
    history = _history_from_returns([-0.01, 0.01, -0.02])
    assert estimate_historical_risk(
        history,
        {"RISK": 10_000.0},
        equity=0.0,
        confidence=0.99,
        lookback_days=100,
        min_observations=2,
    ) is None
    assert estimate_historical_risk(
        history,
        {"RISK": 10_000.0},
        equity=100_000.0,
        confidence=0.99,
        lookback_days=100,
        min_observations=10,
    ) is None


def test_var_gate_allows_sells_and_safe_buys_but_blocks_excess_risk():
    history = _history_from_returns([-0.10, 0.01, -0.08, 0.01, -0.06, 0.02, -0.04, 0.01])
    sell = TradeIntent("OLD", "sell", 5_000.0, "exit")
    safe_buy = TradeIntent("RISK", "buy", 5_000.0, "small")
    large_buy = TradeIntent("RISK", "buy", 50_000.0, "large")

    accepted, diagnostics = filter_buy_intents_by_var(
        [sell, safe_buy, large_buy],
        current_exposures={},
        history=history,
        equity=100_000.0,
        config=_var_config(),
    )

    assert sell in accepted
    assert safe_buy in accepted
    assert large_buy not in accepted
    assert diagnostics.blocked_symbols == ["RISK"]
    assert diagnostics.accepted_buys == 1
    assert diagnostics.blocked_buys == 1


def test_var_gate_fails_closed_when_history_is_missing_and_can_be_disabled():
    buy = TradeIntent("NEW", "buy", 5_000.0, "entry")
    blocked, diagnostics = filter_buy_intents_by_var(
        [buy],
        current_exposures={},
        history={},
        equity=100_000.0,
        config=_var_config(),
    )
    unchanged, disabled = filter_buy_intents_by_var(
        [buy],
        current_exposures={},
        history={},
        equity=100_000.0,
        config={"enabled": False},
    )

    assert blocked == []
    assert diagnostics.blocked_symbols == ["NEW"]
    assert unchanged == [buy]
    assert disabled.blocked_buys == 0


def test_live_risk_manager_applies_var_gate_to_benchmark_core_buy():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.99,
                ("risk", "exit_score_threshold"): -2.0,
                ("risk", "gap_skip_pct"): 0.99,
                ("risk", "sector_caps"): {},
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "value_at_risk"): _var_config(
                    confidence=0.95,
                    min_observations=20,
                    max_var_pct=0.001,
                    max_expected_shortfall_pct=0.0015,
                ),
            }
            return values.get(keys, default)

    class DummyBroker:
        @staticmethod
        def account():
            return Account(100_000.0, 100_000.0, 100_000.0, 100_000.0)

        @staticmethod
        def positions():
            return []

    class DummyState:
        @staticmethod
        def portfolio_guard_tripped():
            return False

        @staticmethod
        def day_start_equity(equity):
            return equity

        @staticmethod
        def in_cooldown(symbol):
            return False

    history = _history_from_returns([-0.03, 0.025] * 50, "QQQ")
    risk = RiskManager(DummyConfig(), DummyBroker(), state=DummyState())

    intents = risk.size_orders(scores={}, prices={"QQQ": 100.0}, history=history)

    assert intents == []


def test_simulator_records_var_diagnostics_and_blocks_oversized_core_buy():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("execution", "fractional_shares"): True,
                ("strategies", "hedge_fund", "enabled"): False,
                ("strategies", "classical", "enabled"): False,
                ("strategies", "ml", "enabled"): False,
                ("strategies", "politicians", "enabled"): False,
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.99,
                ("risk", "exit_score_threshold"): -2.0,
                ("risk", "gap_skip_pct"): 0.99,
                ("risk", "cooldown_days"): 3,
                ("risk", "trailing_activate_pct"): 10.0,
                ("risk", "trailing_giveback_pct"): 1.0,
                ("risk", "earnings_blackout_days"): 0,
                ("risk", "stop_atr_mult"): 100.0,
                ("risk", "stop_min_pct"): 0.99,
                ("risk", "stop_max_pct"): 0.99,
                ("risk", "sector_caps"): {"etf_tech": 3},
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "value_at_risk"): _var_config(
                    confidence=0.95,
                    min_observations=20,
                    max_var_pct=0.001,
                    max_expected_shortfall_pct=0.0015,
                ),
                ("data", "history_days"): 100,
            }
            return values.get(keys, default)

    returns = [-0.03, 0.025] * 50
    history = _history_from_returns(returns, "QQQ")
    dates = history["QQQ"].index
    result = simulate_current_bot(
        DummyConfig(),
        history,
        start_date=dates[60],
        start_capital=100_000.0,
        cost_bps=0.0,
    )

    assert result.summary["var_enabled"] is True
    assert result.summary["var_blocked_buys"] > 0
    assert result.summary["buys"] == 0
    assert {"historical_var_pct", "expected_shortfall_pct"}.issubset(result.equity_curve.columns)
