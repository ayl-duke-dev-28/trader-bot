"""Tests for macro and dynamic sector-risk overlays on equity trades."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.broker.alpaca_client import Account
from src.risk.manager import RiskManager
from src.risk.sector import calculate_sector_risk


def _close_history(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(values), freq="B")
    close = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


def _sector_config(**overrides):
    config = {
        "enabled": True,
        "sma_window": 20,
        "volatility_window": 20,
        "min_members": 2,
        "min_breadth": 0.50,
        "weak_breadth_multiplier": 0.50,
        "max_annualized_vol": 0.30,
        "min_multiplier": 0.25,
    }
    config.update(overrides)
    return config


def test_sector_risk_combines_breadth_and_realized_volatility():
    rising = np.linspace(100.0, 125.0, 60).tolist()
    falling = np.linspace(125.0, 90.0, 60).tolist()
    volatile = (100.0 * np.exp(np.cumsum([0.04, -0.04] * 30))).tolist()
    history = {
        "AAPL": _close_history(falling),
        "MSFT": _close_history(falling),
        "AMD": _close_history(rising),
        "AVGO": _close_history(volatile),
    }

    risks = calculate_sector_risk(history, _sector_config())

    assert risks["mega_cap_tech"].breadth == 0.0
    assert risks["mega_cap_tech"].multiplier == 0.50
    assert risks["ai_infra"].breadth == 1.0
    assert risks["ai_infra"].annualized_volatility > 0.30
    assert 0.25 <= risks["ai_infra"].multiplier < 1.0


def test_sector_risk_is_neutral_when_too_few_members_have_history():
    risks = calculate_sector_risk(
        {"AAPL": _close_history(np.linspace(100.0, 120.0, 60).tolist())},
        _sector_config(min_members=2),
    )

    assert "mega_cap_tech" not in risks


def test_live_risk_manager_caps_gross_exposure_for_macro_contraction():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "macro_cycle"):
                return {
                    "enabled": True,
                    "neutral_max_gross_exposure": 0.60,
                    "contraction_max_gross_exposure": 0.30,
                }
            return default

    cycles = pd.DataFrame(
        {
            "long_score": [-0.7],
            "short_score": [-0.8],
            "composite_score": [-0.74],
            "regime": ["contraction"],
        },
        index=[pd.Timestamp("2026-06-30")],
    )
    risk = RiskManager(DummyConfig(), broker=object(), state=object())

    assert risk._macro_adjusted_max_gross_pct(
        cycles,
        as_of=pd.Timestamp("2026-07-31"),
        current_max_gross_pct=0.80,
    ) == 0.30
    # A stricter market trend cap remains authoritative.
    assert risk._macro_adjusted_max_gross_pct(
        cycles,
        as_of=pd.Timestamp("2026-07-31"),
        current_max_gross_pct=0.20,
    ) == 0.20


def test_live_buy_size_is_reduced_in_a_weak_sector():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.55,
                ("risk", "exit_score_threshold"): 0.0,
                ("risk", "gap_skip_pct"): 0.99,
                ("risk", "sector_caps"): {"mega_cap_tech": 5},
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "macro_cycle"): {"enabled": False},
                ("risk", "benchmark_core"): {"enabled": False},
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "sector_risk"): _sector_config(max_annualized_vol=10.0),
                ("risk", "value_at_risk"): {"enabled": False},
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

    falling = np.linspace(125.0, 90.0, 60).tolist()
    history = {
        "AAPL": _close_history(falling),
        "MSFT": _close_history(falling),
    }
    risk = RiskManager(DummyConfig(), DummyBroker(), state=DummyState())

    intents = risk.size_orders(
        scores={"AAPL": 1.0},
        prices={"AAPL": 90.0},
        history=history,
    )

    assert len(intents) == 1
    assert intents[0].symbol == "AAPL"
    assert intents[0].target_dollars == 2_500.0
    assert "sector_risk=0.50" in intents[0].reason
