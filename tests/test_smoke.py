"""Lightweight smoke tests that don't require network or Alpaca keys."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.classical import classical_signal
from src.signals.hedge_fund import hedge_fund_decision
from src.signals.ml import build_features, build_training_set
from src.signals.momentum_breakout import momentum_breakout_scores
from src.backtest.engine import backtest
from src.broker.alpaca_client import Position
from src.risk.manager import RiskManager, TradeIntent
from src.risk.state import RiskState
from src.trader import _consolidate_intents, _execution_qty_price, _next_scheduled_run


def _fake_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.003, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_classical_signal_in_range():
    s = classical_signal(_fake_df())
    assert -1.0 <= s <= 1.0


def test_build_features_shape():
    feats = build_features(_fake_df()).dropna()
    assert not feats.empty
    assert "rsi_14" in feats.columns


def test_training_set_is_chronological_across_symbols():
    a = _fake_df()
    b = _fake_df()
    b.index = b.index + pd.Timedelta(days=30)
    horizon_days = 5
    X, y = build_training_set({"BBB": b, "AAA": a}, horizon_days=horizon_days)
    assert not X.empty
    assert len(X) == len(y)
    assert X.index.is_monotonic_increasing
    assert X.index.max() <= b.index[-horizon_days - 1]


def test_hedge_fund_signal_in_range():
    class DummyConfig:
        def get(self, *keys, default=None):
            return default

    decision = hedge_fund_decision(DummyConfig(), _fake_df(), bundle=None)
    assert -1.0 <= decision.score <= 1.0
    assert decision.signal in {"bullish", "bearish", "neutral"}
    assert decision.votes


def test_momentum_breakout_selects_top_prior_winner():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("strategies", "momentum_breakout"):
                return {
                    "enabled": True,
                    "top_n": 1,
                    "lookback_days": 60,
                    "min_return": 0.50,
                    "sma_window": 20,
                    "volatility_window": 10,
                    "max_annualized_vol": 10.0,
                    "benchmark_symbol": "QQQ",
                    "benchmark_sma_window": 20,
                    "exclude_symbols": [],
                }
            return default

    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    qqq = pd.DataFrame({"close": np.linspace(100.0, 120.0, len(idx)), "volume": 1_000_000}, index=idx)
    winner = pd.DataFrame({"close": np.linspace(10.0, 30.0, len(idx)), "volume": 1_000_000}, index=idx)
    laggard = pd.DataFrame({"close": np.linspace(10.0, 18.0, len(idx)), "volume": 1_000_000}, index=idx)

    scores = momentum_breakout_scores(DummyConfig(), {"QQQ": qqq, "WIN": winner, "LAG": laggard})
    assert scores["WIN"] == 1.0
    assert scores["LAG"] == 0.0


def test_intent_to_qty_whole_share_mode():
    intent = TradeIntent("MKSI", "buy", 399.59, "score=0.40")
    assert RiskManager.intent_to_qty(intent, 390.99, allow_fractional=False) == 1.0
    assert RiskManager.intent_to_qty(intent, 390.99, allow_fractional=True) == 1.022


def test_consolidate_duplicate_buy_intents():
    intents = [
        TradeIntent("MKSI", "buy", 390.99, "score=0.30"),
        TradeIntent("MKSI", "buy", 1_954.95, "score=0.50"),
        TradeIntent("NVDA", "buy", 500.0, "score=0.40"),
    ]
    merged = _consolidate_intents(intents)
    assert len(merged) == 2
    assert merged[0].symbol == "MKSI"
    assert merged[0].target_dollars == 2_345.94


def test_sell_execution_uses_position_qty_when_quote_missing():
    intent = TradeIntent("C", "sell", 2_113.0, "score=+0.00 <= exit_thr=+0.00")
    position = Position("C", qty=25.0, avg_entry_price=80.0, market_value=2_125.0, unrealized_plpc=0.05)
    qty, price = _execution_qty_price(intent, prices={}, positions={"C": position}, allow_fractional=False)
    assert qty == 25.0
    assert price == 85.0


def test_market_regime_reduces_gross_exposure():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "market_regime"):
                return {
                    "enabled": True,
                    "benchmark_symbol": "QQQ",
                    "sma_window": 3,
                    "risk_off_max_gross_exposure": 0.2,
                }
            return default

    class DummyBroker:
        pass

    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    qqq = pd.DataFrame({"close": [100.0, 99.0, 98.0, 90.0]}, index=idx)
    risk = RiskManager(DummyConfig(), DummyBroker(), state=object())
    assert risk._regime_adjusted_max_gross_pct({"QQQ": qqq}, 0.8) == 0.2


def test_risk_state_tracks_portfolio_guard():
    with TemporaryDirectory() as tmp:
        state = RiskState(Path(tmp) / "risk_state.json")
        assert state.portfolio_highwater(100_000.0) == 100_000.0
        assert state.portfolio_highwater(90_000.0) == 100_000.0
        assert state.portfolio_highwater(110_000.0) == 110_000.0
        assert not state.portfolio_guard_tripped()
        state.trip_portfolio_guard()
        assert state.portfolio_guard_tripped()


def test_benchmark_core_buy_targets_configured_sleeve():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "benchmark_core"):
                return {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.30,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 500,
                }
            return default

    class DummyBroker:
        pass

    risk = RiskManager(DummyConfig(), DummyBroker(), state=object())
    intent = risk._benchmark_core_buy(
        held_active={},
        prices={"QQQ": 100.0},
        equity=100_000.0,
        remaining_gross=80_000.0,
        max_gross_pct=0.8,
        normal_max_gross_pct=0.8,
        open_slots=1,
    )
    assert intent is not None
    assert intent.symbol == "QQQ"
    assert intent.target_dollars == 30_000.0


def test_relative_strength_blocks_lagging_symbol():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "relative_strength"):
                return {
                    "enabled": True,
                    "benchmark_symbol": "QQQ",
                    "lookback_days": 3,
                    "min_excess_return": 0.0,
                    "exempt_symbols": ["QQQ"],
                }
            return default

    class DummyBroker:
        pass

    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    history = {
        "AAPL": pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 101.0]}, index=idx),
        "QQQ": pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 110.0]}, index=idx),
    }
    risk = RiskManager(DummyConfig(), DummyBroker(), state=object())
    assert not risk._passes_relative_strength("AAPL", history)
    assert risk._passes_relative_strength("QQQ", history)


def test_backtest_uses_live_path_benchmark_core():
    class DummyConfig:
        is_live = False

        def get(self, *keys, default=None):
            values = {
                ("execution", "fractional_shares"): True,
                ("strategies", "hedge_fund", "enabled"): False,
                ("strategies", "classical", "enabled"): False,
                ("strategies", "classical", "weight"): 0.0,
                ("strategies", "ml", "enabled"): False,
                ("strategies", "ml", "weight"): 0.0,
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
                ("risk", "earnings_blackout_days"): 3,
                ("risk", "stop_atr_mult"): 100.0,
                ("risk", "stop_min_pct"): 0.99,
                ("risk", "stop_max_pct"): 0.99,
                ("risk", "sector_caps"): {"etf_tech": 3, "other": 3},
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("data", "history_days"): 40,
                ("backtest", "warmup_days"): 0,
            }
            return values.get(keys, default)

    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    qqq = pd.DataFrame(
        {
            "open": np.linspace(100.0, 120.0, len(idx)),
            "high": np.linspace(101.0, 121.0, len(idx)),
            "low": np.linspace(99.0, 119.0, len(idx)),
            "close": np.linspace(100.0, 120.0, len(idx)),
            "volume": 1_000_000,
        },
        index=idx,
    )

    result = backtest(
        DummyConfig(),
        {"QQQ": qqq},
        start_date=idx[60],
        start_capital=100_000.0,
        cost_bps=0.0,
    )

    assert result.summary is not None
    assert result.trades_log is not None
    assert result.summary["buys"] >= 1
    assert result.summary["profit_days"] >= 0
    assert result.summary["loss_days"] >= 0
    assert 0.0 <= result.summary["loss_day_rate"] <= 1.0
    assert "worst_day_return" in result.summary
    assert "benchmark core target=50%" in set(result.trades_log["reason"])

    blocked = backtest(
        DummyConfig(),
        {"QQQ": qqq},
        start_date=idx[60],
        start_capital=100_000.0,
        cost_bps=0.0,
        earnings_calendar={"QQQ": [d for d in idx[60:]]},
    )
    assert blocked.summary is not None
    assert blocked.summary["buys"] == 0


def test_next_scheduled_run_uses_market_hours_et():
    tz = ZoneInfo("America/New_York")
    assert _next_scheduled_run(datetime(2026, 7, 9, 8, 0, tzinfo=tz)) == datetime(2026, 7, 9, 9, 30, tzinfo=tz)
    assert _next_scheduled_run(datetime(2026, 7, 9, 9, 31, tzinfo=tz)) == datetime(2026, 7, 9, 10, 30, tzinfo=tz)
    assert _next_scheduled_run(datetime(2026, 7, 9, 15, 30, tzinfo=tz)) == datetime(2026, 7, 9, 15, 30, tzinfo=tz)
    assert _next_scheduled_run(datetime(2026, 7, 9, 15, 31, tzinfo=tz)) == datetime(2026, 7, 10, 9, 30, tzinfo=tz)
    assert _next_scheduled_run(datetime(2026, 7, 10, 16, 0, tzinfo=tz)) == datetime(2026, 7, 13, 9, 30, tzinfo=tz)


if __name__ == "__main__":
    test_classical_signal_in_range()
    test_build_features_shape()
    test_training_set_is_chronological_across_symbols()
    test_hedge_fund_signal_in_range()
    test_momentum_breakout_selects_top_prior_winner()
    test_intent_to_qty_whole_share_mode()
    test_consolidate_duplicate_buy_intents()
    test_sell_execution_uses_position_qty_when_quote_missing()
    test_market_regime_reduces_gross_exposure()
    test_risk_state_tracks_portfolio_guard()
    test_benchmark_core_buy_targets_configured_sleeve()
    test_relative_strength_blocks_lagging_symbol()
    test_backtest_uses_live_path_benchmark_core()
    test_next_scheduled_run_uses_market_hours_et()
    print("smoke tests OK")
