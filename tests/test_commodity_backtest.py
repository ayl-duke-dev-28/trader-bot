"""Tests for the leakage-safe commodity walk-forward backtest."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.commodities import (
    CommodityParameters,
    build_walk_forward_windows,
    commodity_target_weights,
    default_diversified_candidates,
    run_commodity_walk_forward,
)


def _synthetic_prices() -> pd.DataFrame:
    index = pd.date_range("2008-01-02", "2022-12-30", freq="B")
    step = np.arange(len(index), dtype=float)
    # LEADER changes after the training period. A leakage-free run cannot use
    # that future reversal while selecting the first test window's parameters.
    leader = 100.0 * np.exp(0.00045 * step)
    diversifier = 80.0 * np.exp(0.00020 * step + 0.04 * np.sin(step / 40.0))
    loser = 120.0 * np.exp(-0.00020 * step)
    return pd.DataFrame(
        {"LEADER": leader, "DIVERSIFIER": diversifier, "LOSER": loser},
        index=index,
    )


class CommodityWalkForwardTests(unittest.TestCase):
    def test_windows_use_rolling_prior_five_years_and_four_calendar_month_tests(self):
        prices = _synthetic_prices()

        windows = build_walk_forward_windows(
            prices.index,
            train_years=5,
            test_months=4,
        )

        self.assertGreaterEqual(len(windows), 20)
        first = windows[0]
        self.assertLess(first.train_end, first.test_start)
        self.assertEqual(first.test_end, first.test_start + pd.DateOffset(months=4) - pd.Timedelta(days=1))
        self.assertEqual(windows[1].test_start, first.test_start + pd.DateOffset(months=4))
        for window in windows:
            self.assertLessEqual(window.train_start, window.train_end)
            self.assertLess(window.train_end, window.test_start)

    def test_target_weights_hold_cash_when_no_asset_has_positive_absolute_momentum(self):
        index = pd.date_range("2020-01-01", periods=300, freq="B")
        prices = pd.DataFrame(
            {
                "A": np.linspace(100.0, 70.0, len(index)),
                "B": np.linspace(90.0, 60.0, len(index)),
            },
            index=index,
        )

        weights = commodity_target_weights(
            prices,
            CommodityParameters(momentum_days=126, top_n=2),
        )

        self.assertAlmostEqual(float(weights.sum()), 0.0)

    def test_target_weights_are_long_only_capped_and_prefer_the_stronger_trend(self):
        prices = _synthetic_prices().iloc[:1000]

        weights = commodity_target_weights(
            prices,
            CommodityParameters(momentum_days=126, top_n=2, max_position=0.60),
        )

        self.assertTrue((weights >= 0.0).all())
        self.assertLessEqual(float(weights.sum()), 1.0 + 1e-12)
        self.assertLessEqual(float(weights.max()), 0.60 + 1e-12)
        self.assertGreater(float(weights["LEADER"]), float(weights["LOSER"]))

    def test_diversified_weights_cap_positions_and_correlated_commodity_groups(self):
        index = pd.date_range("2018-01-01", periods=320, freq="B")
        step = np.arange(len(index), dtype=float)
        prices = pd.DataFrame(
            {
                "USO": 50 * np.exp(0.0012 * step),
                "BNO": 50 * np.exp(0.0011 * step),
                "UNG": 50 * np.exp(0.0010 * step),
                "GLD": 50 * np.exp(0.0008 * step),
                "DBA": 50 * np.exp(0.0007 * step),
                "CPER": 50 * np.exp(0.0006 * step),
            },
            index=index,
        )
        parameters = default_diversified_candidates()[0]

        weights = commodity_target_weights(prices, parameters)

        self.assertLessEqual(float(weights.max()), 0.25 + 1e-12)
        self.assertLessEqual(float(weights[["USO", "BNO", "UNG"]].sum()), 0.35 + 1e-12)
        self.assertGreaterEqual(int((weights > 0).sum()), 4)
        self.assertLessEqual(float(weights.sum()), 1.0 + 1e-12)

    def test_walk_forward_selection_never_receives_test_or_future_prices(self):
        prices = _synthetic_prices()
        observations: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def selector(train_prices, candidates, cost_bps):
            observations.append((train_prices.index.max(), train_prices.attrs["test_start"]))
            return candidates[0], {"train_sharpe": 0.0}

        result = run_commodity_walk_forward(
            prices,
            candidates=[CommodityParameters(momentum_days=63, top_n=1)],
            train_years=5,
            test_months=4,
            cost_bps=10.0,
            parameter_selector=selector,
        )

        self.assertGreater(len(result.windows), 20)
        self.assertTrue(all(train_end < test_start for train_end, test_start in observations))
        self.assertEqual(result.summary["walk_forward_windows"], len(result.windows))
        self.assertTrue((result.windows["test_observations"] > 0).all())
        self.assertTrue(np.isfinite(result.windows["test_return"]).all())
        self.assertTrue(np.isfinite(result.summary["sharpe"]))
        self.assertTrue(np.isfinite(result.summary["max_drawdown"]))
        self.assertAlmostEqual(
            result.summary["total_return"],
            result.summary["final_equity"] / result.summary["start_capital"] - 1.0,
        )

    def test_transaction_costs_reduce_walk_forward_equity(self):
        prices = _synthetic_prices()
        candidate = [CommodityParameters(momentum_days=63, top_n=1)]

        free = run_commodity_walk_forward(prices, candidate, cost_bps=0.0)
        costly = run_commodity_walk_forward(prices, candidate, cost_bps=50.0)

        self.assertLess(costly.equity_curve.iloc[-1], free.equity_curve.iloc[-1])
        self.assertGreater(costly.summary["transaction_costs"], 0.0)


if __name__ == "__main__":
    unittest.main()
