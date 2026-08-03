"""Lightweight smoke tests that don't require network or Alpaca keys."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.classical import classical_signal
from src.signals.hedge_fund import hedge_fund_decision
from src.signals.ml import build_features, build_training_set
from src.signals.momentum_breakout import momentum_breakout_scores
from src.backtest.engine import backtest
from src.broker.alpaca_client import Account, AlpacaBroker, Position, _retry_request
from src.risk.manager import RiskManager, TradeIntent
from src.risk.state import RiskState
from src.risk.validation import is_valid_price
from src.trader import _consolidate_intents, _execution_qty_price, _last_prices, _next_scheduled_run


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


def test_intent_to_qty_rejects_non_finite_prices():
    intent = TradeIntent("BAD", "buy", 1_000.0, "score=0.80")
    for price in (float("nan"), float("inf"), float("-inf")):
        assert RiskManager.intent_to_qty(intent, price, allow_fractional=False) == 0.0
        assert RiskManager.intent_to_qty(intent, price, allow_fractional=True) == 0.0


def test_price_validation_rejects_non_numeric_and_non_positive_values():
    assert is_valid_price(100.0)
    assert not is_valid_price(None)
    assert not is_valid_price("not-a-price")
    assert not is_valid_price(0.0)
    assert not is_valid_price(-1.0)
    assert not is_valid_price(float("nan"))
    assert not is_valid_price(float("inf"))


def test_last_prices_drops_non_finite_quotes():
    columns = pd.MultiIndex.from_product(
        [["Close"], ["GOOD", "NAN", "INF", "ZERO", "MISSING", "TEXT"]]
    )
    download = pd.DataFrame(
        [[100.0, float("nan"), float("inf"), 0.0, pd.NA, "bad"]],
        columns=columns,
        index=[pd.Timestamp("2026-07-29")],
    )

    with patch("src.trader.yf.download", return_value=download):
        prices = _last_prices(["GOOD", "NAN", "INF", "ZERO", "MISSING", "TEXT"])

    assert prices == {"GOOD": 100.0}


def test_last_prices_recovers_invalid_quotes_with_fallback():
    columns = pd.MultiIndex.from_product([["Close"], ["GOOD", "RECOVER", "UNRESOLVED"]])
    download = pd.DataFrame(
        [[100.0, float("nan"), float("nan")]],
        columns=columns,
        index=[pd.Timestamp("2026-07-30")],
    )
    fallback_calls: list[list[str]] = []

    def fallback(symbols: list[str]) -> dict[str, float]:
        fallback_calls.append(symbols)
        return {"RECOVER": 42.5, "UNRESOLVED": float("nan")}

    with patch("src.trader.yf.download", return_value=download):
        prices = _last_prices(["GOOD", "RECOVER", "UNRESOLVED"], fallback=fallback)

    assert fallback_calls == [["RECOVER", "UNRESOLVED"]]
    assert prices == {"GOOD": 100.0, "RECOVER": 42.5}


def test_last_prices_recovers_all_symbols_when_yahoo_fails():
    fallback = MagicMock(return_value={"GOOD": 100.0})

    with patch("src.trader.yf.download", side_effect=RuntimeError("Yahoo unavailable")):
        prices = _last_prices(["GOOD"], fallback=fallback)

    fallback.assert_called_once_with(["GOOD"])
    assert prices == {"GOOD": 100.0}


def test_last_prices_handles_empty_input_and_fallback_failure():
    assert _last_prices([], fallback=MagicMock()) == {}

    columns = pd.MultiIndex.from_product([["Close"], ["BAD"]])
    download = pd.DataFrame(
        [[float("nan")]],
        columns=columns,
        index=[pd.Timestamp("2026-07-30")],
    )
    with patch("src.trader.yf.download", return_value=download):
        prices = _last_prices(
            ["BAD"],
            fallback=MagicMock(side_effect=RuntimeError("Alpaca unavailable")),
        )

    assert prices == {}


def test_alpaca_latest_prices_uses_iex_and_filters_invalid_trades():
    class Trade:
        def __init__(self, price):
            self.price = price

    class DataClient:
        request = None

        def get_stock_latest_trade(self, request):
            self.request = request
            return {
                "GOOD": Trade(101.25),
                "NAN": Trade(float("nan")),
                "NONE": None,
            }

    broker = object.__new__(AlpacaBroker)
    broker.market_data_client = DataClient()

    prices = broker.latest_prices(["GOOD", "NAN", "NONE", "ABSENT"])

    assert broker.market_data_client.request.symbol_or_symbols == ["GOOD", "NAN", "NONE", "ABSENT"]
    assert broker.market_data_client.request.feed.value == "iex"
    assert prices == {"GOOD": 101.25}


def test_alpaca_latest_prices_handles_empty_input_and_api_failure():
    class FailingDataClient:
        @staticmethod
        def get_stock_latest_trade(request):
            raise ValueError("bad response")

    broker = object.__new__(AlpacaBroker)
    broker.market_data_client = FailingDataClient()

    assert broker.latest_prices([]) == {}
    assert broker.latest_prices(["BAD"]) == {}


def test_trade_once_passes_alpaca_price_fallback_to_quote_loader():
    class DummyConfig:
        is_live = False

        @staticmethod
        def get(*keys, default=None):
            return default

    broker = MagicMock()
    broker.is_market_open.return_value = True
    broker.positions.return_value = []
    risk = MagicMock()
    risk.apply_portfolio_drawdown_guard.return_value = False
    risk.size_orders.return_value = []
    macro_cycles = pd.DataFrame(
        {"regime": ["neutral"]},
        index=[pd.Timestamp("2026-06-30")],
    )

    with (
        patch("src.trader.AlpacaBroker", return_value=broker),
        patch("src.trader.trade_logger_from_config"),
        patch("src.trader.RiskManager", return_value=risk),
        patch("src.trader.load_universe", return_value=["GOOD"]),
        patch("src.trader._history_for_all", return_value={}),
        patch("src.trader.compute_signals", return_value={"GOOD": 1.0}),
        patch("src.trader._load_live_macro_cycles", return_value=macro_cycles),
        patch("src.trader._last_prices", return_value={"GOOD": 100.0}) as price_loader,
    ):
        from src.trader import trade_once

        trade_once(DummyConfig())

    price_loader.assert_called_once_with(["GOOD"], fallback=broker.latest_prices)
    assert risk.size_orders.call_args.kwargs["macro_cycles"] is macro_cycles


def test_live_trade_cycle_routes_cap_trim_as_partial_sell_order():
    class DummyConfig:
        is_live = False

        @staticmethod
        def get(*keys, default=None):
            values = {
                ("dry_run",): False,
                ("execution", "fractional_shares"): False,
                ("risk", "earnings_blackout_days"): 0,
            }
            return values.get(keys, default)

    position = Position(
        "AAPL",
        qty=200.0,
        avg_entry_price=80.0,
        market_value=20_000.0,
        unrealized_plpc=0.25,
    )
    broker = MagicMock()
    broker.is_market_open.return_value = True
    broker.positions.return_value = [position]
    pending_sells: set[str] = set()
    broker.open_order_symbols.side_effect = (
        lambda side=None: pending_sells if side == "sell" else set()
    )
    broker.submit_market_order.return_value = "trim-order-id"
    risk = MagicMock()
    risk.apply_portfolio_drawdown_guard.return_value = False
    risk.size_orders.return_value = [
        TradeIntent(
            "AAPL",
            "sell",
            10_000.0,
            "gross cap rebalance",
            sell_entire_position=False,
        )
    ]
    trade_log = MagicMock()

    with (
        patch("src.trader.AlpacaBroker", return_value=broker),
        patch("src.trader.trade_logger_from_config", return_value=trade_log),
        patch("src.trader.RiskManager", return_value=risk),
        patch("src.trader.load_universe", return_value=["AAPL"]),
        patch("src.trader._history_for_all", return_value={}),
        patch("src.trader.compute_signals", return_value={"AAPL": 0.10}),
        patch("src.trader._load_live_macro_cycles", return_value=None),
        patch("src.trader._last_prices", return_value={"AAPL": 100.0}),
    ):
        from src.trader import trade_once

        trade_once(DummyConfig())
        broker.submit_market_order.assert_called_once_with("AAPL", 100.0, "sell")
        broker.close_position.assert_not_called()
        assert broker.open_order_symbols.call_args_list == [
            call(side="buy"),
            call(side="sell"),
        ]
        risk.state.clear_symbol.assert_not_called()
        assert trade_log.log.call_args.args[0].action == "TRIM"

        pending_sells.add("AAPL")
        broker.submit_market_order.reset_mock()
        broker.open_order_symbols.reset_mock()
        trade_log.log.reset_mock()
        trade_once(DummyConfig())

        broker.submit_market_order.assert_not_called()
        broker.close_position.assert_not_called()
        assert trade_log.log.call_args.args[0].action == "SKIP"
        assert "sell already pending" in trade_log.log.call_args.args[0].reason


def test_size_orders_skips_non_finite_quote():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "benchmark_core"): {"enabled": False},
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "portfolio_drawdown_guard"): {"enabled": False},
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

    risk = RiskManager(DummyConfig(), DummyBroker(), state=DummyState())
    intents = risk.size_orders(
        scores={"BAD": 0.80},
        prices={"BAD": float("nan")},
        history={},
    )

    assert intents == []


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


def test_execution_qty_rejects_nan_buy_and_falls_back_for_sell():
    buy = TradeIntent("BAD", "buy", 1_000.0, "score=0.80")
    buy_qty, buy_price = _execution_qty_price(
        buy,
        prices={"BAD": float("nan")},
        positions={},
        allow_fractional=False,
    )
    assert buy_qty == 0.0
    assert buy_price == 0.0

    sell = TradeIntent("BAD", "sell", 2_125.0, "score=0.00")
    position = Position("BAD", qty=25.0, avg_entry_price=80.0, market_value=2_125.0, unrealized_plpc=0.05)
    sell_qty, sell_price = _execution_qty_price(
        sell,
        prices={"BAD": float("nan")},
        positions={"BAD": position},
        allow_fractional=False,
    )
    assert sell_qty == 25.0
    assert sell_price == 85.0


def test_partial_sell_execution_uses_target_dollars_instead_of_closing_position():
    intent = TradeIntent(
        "AAPL",
        "sell",
        10_000.0,
        "gross cap rebalance",
        sell_entire_position=False,
    )
    position = Position(
        "AAPL",
        qty=200.0,
        avg_entry_price=80.0,
        market_value=20_000.0,
        unrealized_plpc=0.25,
    )

    qty, price = _execution_qty_price(
        intent,
        prices={"AAPL": 100.0},
        positions={"AAPL": position},
        allow_fractional=False,
    )

    assert qty == 100.0
    assert price == 100.0


def test_active_gross_cap_creates_only_the_required_partial_sell():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.55,
                ("risk", "exit_score_threshold"): 0.0,
                ("risk", "gap_skip_pct"): 0.99,
                ("risk", "sector_caps"): {"mega_cap_tech": 5, "etf_tech": 3},
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "macro_cycle"): {
                    "enabled": True,
                    "neutral_max_gross_exposure": 0.60,
                    "contraction_max_gross_exposure": 0.30,
                },
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "neutral_target_pct": 0.50,
                    "contraction_target_pct": 0.0,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "sector_risk"): {"enabled": False},
                ("risk", "value_at_risk"): {"enabled": False},
            }
            return values.get(keys, default)

    class DummyBroker:
        @staticmethod
        def account():
            return Account(100_000.0, 30_000.0, 30_000.0, 100_000.0)

        @staticmethod
        def positions():
            return [
                Position("QQQ", 500.0, 80.0, 50_000.0, 0.25),
                Position("AAPL", 200.0, 80.0, 20_000.0, 0.25),
            ]

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

    as_of = pd.Timestamp("2026-07-31")
    macro_cycles = pd.DataFrame(
        {
            "long_score": [-0.10],
            "short_score": [-0.10],
            "composite_score": [-0.10],
            "regime": ["neutral"],
        },
        index=[as_of],
    )
    risk = RiskManager(DummyConfig(), DummyBroker(), state=DummyState())

    intents = risk.size_orders(
        scores={"QQQ": 0.10, "AAPL": 0.10},
        prices={"QQQ": 100.0, "AAPL": 100.0},
        history={},
        macro_cycles=macro_cycles,
        as_of=as_of,
    )

    assert len(intents) == 1
    assert intents[0].symbol == "AAPL"
    assert intents[0].side == "sell"
    assert intents[0].target_dollars == 10_000.0
    assert intents[0].sell_entire_position is False
    assert "gross cap" in intents[0].reason


def test_gross_cap_trims_core_growth_above_target_before_other_holdings():
    class DummyConfig:
        @staticmethod
        def get(*keys, default=None):
            return default

    risk = RiskManager(DummyConfig(), broker=object(), state=object())
    intents: list[TradeIntent] = []
    held = {
        "QQQ": Position("QQQ", 550.0, 80.0, 55_000.0, 0.25),
        "AAPL": Position("AAPL", 100.0, 80.0, 10_000.0, 0.25),
    }

    risk._apply_gross_cap_sells(
        intents=intents,
        held=held,
        scores={"QQQ": 0.50, "AAPL": -0.50},
        equity=100_000.0,
        max_gross_pct=0.60,
        core_symbol="QQQ",
        core_target_pct=0.50,
    )

    assert len(intents) == 1
    assert intents[0].symbol == "QQQ"
    assert intents[0].target_dollars == 5_000.0
    assert intents[0].sell_entire_position is False


def test_gross_cap_accounts_for_full_exits_already_planned():
    class DummyConfig:
        @staticmethod
        def get(*keys, default=None):
            return default

    risk = RiskManager(DummyConfig(), broker=object(), state=object())
    existing = TradeIntent("AAPL", "sell", 20_000.0, "score exit")
    intents = [existing]

    risk._apply_gross_cap_sells(
        intents=intents,
        held={
            "QQQ": Position("QQQ", 500.0, 80.0, 50_000.0, 0.25),
            "AAPL": Position("AAPL", 200.0, 80.0, 20_000.0, 0.25),
        },
        scores={"QQQ": 0.50, "AAPL": -0.50},
        equity=100_000.0,
        max_gross_pct=0.60,
        core_symbol="QQQ",
        core_target_pct=0.50,
    )

    assert intents == [existing]


def test_alpaca_read_retry_recovers_from_connection_reset():
    calls = {"count": 0}

    def flaky_read():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RequestsConnectionError("connection reset by peer")
        return "ok"

    assert _retry_request("test read", flaky_read, attempts=2, delay_seconds=0.0) == "ok"
    assert calls["count"] == 2


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

    class DummyState:
        @staticmethod
        def in_cooldown(symbol):
            return False

    risk = RiskManager(DummyConfig(), DummyBroker(), state=DummyState())
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


def test_benchmark_core_honors_stop_cooldown():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "benchmark_core"):
                return {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 500,
                }
            return default

    class CooldownState:
        @staticmethod
        def in_cooldown(symbol):
            return symbol == "QQQ"

    risk = RiskManager(DummyConfig(), broker=object(), state=CooldownState())
    intent = risk._benchmark_core_buy(
        held_active={},
        prices={"QQQ": 680.0},
        equity=100_000.0,
        remaining_gross=80_000.0,
        max_gross_pct=0.8,
        normal_max_gross_pct=0.8,
        open_slots=1,
    )
    assert intent is None


def test_risk_on_benchmark_core_ignores_negative_score_exit():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.55,
                ("risk", "exit_score_threshold"): 0.0,
                ("risk", "gap_skip_pct"): 0.05,
                ("risk", "sector_caps"): {"etf_tech": 3},
                ("risk", "market_regime"): {
                    "enabled": True,
                    "benchmark_symbol": "QQQ",
                    "sma_window": 20,
                    "risk_off_max_gross_exposure": 0.20,
                },
                ("risk", "macro_cycle"): {"enabled": False},
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "sector_risk"): {"enabled": False},
                ("risk", "value_at_risk"): {"enabled": False},
            }
            return values.get(keys, default)

    class DummyBroker:
        @staticmethod
        def account():
            return Account(100_000.0, 100_000.0, 50_000.0, 50_000.0)

        @staticmethod
        def positions():
            return [Position("QQQ", 500.0, 100.0, 50_000.0, 0.0)]

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

    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    qqq = pd.DataFrame({"close": np.linspace(100.0, 130.0, len(idx))}, index=idx)
    risk = RiskManager(DummyConfig(), DummyBroker(), state=DummyState())

    intents = risk.size_orders(
        scores={"QQQ": -0.20},
        prices={"QQQ": 100.0},
        history={"QQQ": qqq},
    )

    assert intents == []


def test_benchmark_core_still_exits_when_regime_target_is_zero():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "benchmark_core"):
                return {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                }
            return default

    risk = RiskManager(DummyConfig(), broker=object(), state=object())
    intents = []

    risk._apply_benchmark_core_sells(
        intents=intents,
        held={"QQQ": Position("QQQ", 500.0, 100.0, 50_000.0, 0.0)},
        max_gross_pct=0.20,
        normal_max_gross_pct=0.80,
    )

    assert len(intents) == 1
    assert intents[0].side == "sell"
    assert intents[0].reason == "benchmark core risk-off target=0"


def test_benchmark_core_target_distinguishes_market_and_macro_regimes():
    class DummyConfig:
        def get(self, *keys, default=None):
            if keys == ("risk", "benchmark_core"):
                return {
                    "enabled": True,
                    "risk_on_target_pct": 0.50,
                    "neutral_target_pct": 0.50,
                    "contraction_target_pct": 0.0,
                    "risk_off_target_pct": 0.0,
                }
            return default

    risk = RiskManager(DummyConfig(), broker=object(), state=object())

    assert risk._benchmark_core_target_pct(
        0.60,
        0.80,
        market_risk_on=True,
        macro_regime="neutral",
    ) == 0.50
    assert risk._benchmark_core_target_pct(
        0.30,
        0.80,
        market_risk_on=True,
        macro_regime="contraction",
    ) == 0.0
    assert risk._benchmark_core_target_pct(
        0.20,
        0.80,
        market_risk_on=False,
        macro_regime="neutral",
    ) == 0.0


def test_backtest_does_not_churn_risk_on_benchmark_core_on_neutral_score():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("execution", "fractional_shares"): True,
                ("strategies", "momentum_breakout", "enabled"): False,
                ("strategies", "hedge_fund", "enabled"): False,
                ("strategies", "classical", "enabled"): False,
                ("strategies", "classical", "weight"): 0.0,
                ("strategies", "ml", "enabled"): False,
                ("strategies", "ml", "weight"): 0.0,
                ("strategies", "politicians", "enabled"): False,
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.55,
                ("risk", "exit_score_threshold"): 0.0,
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
                ("risk", "macro_cycle"): {"enabled": False},
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "sector_risk"): {"enabled": False},
                ("risk", "value_at_risk"): {"enabled": False},
                ("risk", "portfolio_drawdown_guard"): {"enabled": False},
                ("data", "history_days"): 80,
                ("backtest", "warmup_days"): 0,
            }
            return values.get(keys, default)

    idx = pd.date_range("2025-01-01", periods=90, freq="B")
    close = np.linspace(100.0, 120.0, len(idx))
    qqq = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
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
        walk_forward=False,
    )

    assert result.summary is not None
    assert result.summary["buys"] == 1
    assert result.summary["sells"] == 0

    macro_cycles = pd.DataFrame(
        {
            "long_score": [-0.10],
            "short_score": [-0.10],
            "composite_score": [-0.10],
            "regime": ["neutral"],
        },
        index=[idx[0]],
    )
    neutral_result = backtest(
        DummyConfig(),
        {"QQQ": qqq},
        start_date=idx[60],
        start_capital=100_000.0,
        cost_bps=0.0,
        walk_forward=False,
        macro_cycles=macro_cycles,
        macro_cycle_config={
            "enabled": True,
            "neutral_max_gross_exposure": 0.60,
            "contraction_max_gross_exposure": 0.30,
        },
    )

    assert neutral_result.summary is not None
    assert neutral_result.summary["buys"] == 1
    assert neutral_result.summary["sells"] == 0
    assert neutral_result.summary["macro_min_gross_exposure"] == 0.60


def test_backtest_trims_existing_positions_when_macro_cap_falls():
    class DummyConfig:
        def get(self, *keys, default=None):
            values = {
                ("execution", "fractional_shares"): True,
                ("strategies", "momentum_breakout", "enabled"): False,
                ("strategies", "hedge_fund", "enabled"): True,
                ("strategies", "ml", "enabled"): False,
                ("risk", "max_position_pct"): 0.05,
                ("risk", "max_gross_exposure"): 0.80,
                ("risk", "max_positions"): 20,
                ("risk", "entry_score_threshold"): 0.55,
                ("risk", "exit_score_threshold"): 0.0,
                ("risk", "gap_skip_pct"): 0.99,
                ("risk", "cooldown_days"): 3,
                ("risk", "trailing_activate_pct"): 10.0,
                ("risk", "trailing_giveback_pct"): 1.0,
                ("risk", "earnings_blackout_days"): 0,
                ("risk", "stop_atr_mult"): 100.0,
                ("risk", "stop_min_pct"): 0.99,
                ("risk", "stop_max_pct"): 0.99,
                ("risk", "sector_caps"): {"mega_cap_tech": 20, "etf_tech": 3},
                ("risk", "market_regime"): {"enabled": False},
                ("risk", "macro_cycle"): {"enabled": False},
                ("risk", "benchmark_core"): {
                    "enabled": True,
                    "symbol": "QQQ",
                    "risk_on_target_pct": 0.50,
                    "neutral_target_pct": 0.50,
                    "contraction_target_pct": 0.0,
                    "risk_off_target_pct": 0.0,
                    "min_trade_dollars": 100,
                },
                ("risk", "relative_strength"): {"enabled": False},
                ("risk", "sector_risk"): {"enabled": False},
                ("risk", "value_at_risk"): {"enabled": False},
                ("risk", "portfolio_drawdown_guard"): {"enabled": False},
                ("data", "history_days"): 80,
                ("backtest", "warmup_days"): 0,
            }
            return values.get(keys, default)

    idx = pd.date_range("2025-01-01", periods=90, freq="B")
    symbols = ["QQQ", "AAPL", "MSFT", "GOOG", "META", "AMZN", "ORCL"]
    history = {
        symbol: pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
            },
            index=idx,
        )
        for symbol in symbols
    }
    score_cache = {
        symbol: pd.Series(1.0, index=idx)
        for symbol in symbols
    }
    macro_cycles = pd.DataFrame(
        {
            "long_score": [0.30, -0.10],
            "short_score": [0.30, -0.10],
            "composite_score": [0.30, -0.10],
            "regime": ["expansion", "neutral"],
        },
        index=[idx[0], idx[65]],
    )

    with patch(
        "src.backtest.simulator._precompute_hedge_fund_scores",
        return_value=score_cache,
    ):
        result = backtest(
            DummyConfig(),
            history,
            start_date=idx[60],
            end_date=idx[70],
            start_capital=100_000.0,
            cost_bps=0.0,
            walk_forward=False,
            macro_cycles=macro_cycles,
            macro_cycle_config={
                "enabled": True,
                "neutral_max_gross_exposure": 0.60,
                "contraction_max_gross_exposure": 0.30,
            },
        )

    neutral_curve = result.equity_diagnostics.loc[
        result.equity_diagnostics["macro_regime"] == "neutral"
    ]
    actual_gross = (
        (neutral_curve["equity"] - neutral_curve["cash"])
        / neutral_curve["equity"]
    )
    assert (actual_gross <= 0.600001).all()
    assert (result.trades_log["action"] == "TRIM").any()


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

    macro_cycles = pd.DataFrame(
        {
            "long_score": [-0.6],
            "short_score": [-0.7],
            "composite_score": [-0.64],
            "regime": ["contraction"],
        },
        index=[idx[0]],
    )
    macro_limited = backtest(
        DummyConfig(),
        {"QQQ": qqq},
        start_date=idx[60],
        start_capital=100_000.0,
        cost_bps=0.0,
        macro_cycles=macro_cycles,
        macro_cycle_config={
            "enabled": True,
            "neutral_max_gross_exposure": 0.60,
            "contraction_max_gross_exposure": 0.30,
        },
    )
    assert macro_limited.summary is not None
    assert macro_limited.summary["macro_cycle_enabled"] is True
    assert macro_limited.summary["macro_contraction_days"] > 0
    assert macro_limited.summary["macro_min_gross_exposure"] == 0.30
    assert macro_limited.summary["buys"] == 0
    assert macro_limited.equity_diagnostics is not None
    assert set(
        {
            "macro_regime",
            "macro_long_score",
            "macro_short_score",
            "macro_composite_score",
            "max_gross_exposure",
        }
    ).issubset(macro_limited.equity_diagnostics.columns)


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
    test_intent_to_qty_rejects_non_finite_prices()
    test_price_validation_rejects_non_numeric_and_non_positive_values()
    test_last_prices_drops_non_finite_quotes()
    test_size_orders_skips_non_finite_quote()
    test_consolidate_duplicate_buy_intents()
    test_sell_execution_uses_position_qty_when_quote_missing()
    test_execution_qty_rejects_nan_buy_and_falls_back_for_sell()
    test_alpaca_read_retry_recovers_from_connection_reset()
    test_market_regime_reduces_gross_exposure()
    test_risk_state_tracks_portfolio_guard()
    test_benchmark_core_buy_targets_configured_sleeve()
    test_benchmark_core_honors_stop_cooldown()
    test_relative_strength_blocks_lagging_symbol()
    test_backtest_uses_live_path_benchmark_core()
    test_next_scheduled_run_uses_market_hours_et()
    print("smoke tests OK")
