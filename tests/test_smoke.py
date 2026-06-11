"""Lightweight smoke tests that don't require network or Alpaca keys."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.classical import classical_signal
from src.signals.ml import build_features


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


if __name__ == "__main__":
    test_classical_signal_in_range()
    test_build_features_shape()
    print("smoke tests OK")
