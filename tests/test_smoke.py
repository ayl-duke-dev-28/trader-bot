"""Lightweight smoke tests that don't require network or Alpaca keys."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.classical import classical_signal
from src.signals.hedge_fund import hedge_fund_decision
from src.signals.ml import build_features, build_training_set
from src.risk.manager import RiskManager, TradeIntent
from src.trader import _consolidate_intents


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
    X, y = build_training_set({"BBB": b, "AAA": a}, horizon_days=5)
    assert not X.empty
    assert len(X) == len(y)
    assert X.index.is_monotonic_increasing


def test_hedge_fund_signal_in_range():
    class DummyConfig:
        def get(self, *keys, default=None):
            return default

    decision = hedge_fund_decision(DummyConfig(), _fake_df(), bundle=None)
    assert -1.0 <= decision.score <= 1.0
    assert decision.signal in {"bullish", "bearish", "neutral"}
    assert decision.votes


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


if __name__ == "__main__":
    test_classical_signal_in_range()
    test_build_features_shape()
    test_training_set_is_chronological_across_symbols()
    test_hedge_fund_signal_in_range()
    test_intent_to_qty_whole_share_mode()
    test_consolidate_duplicate_buy_intents()
    test_market_regime_reduces_gross_exposure()
    print("smoke tests OK")
