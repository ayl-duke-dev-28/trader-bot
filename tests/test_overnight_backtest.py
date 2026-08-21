import math

import pandas as pd
import pytest

from src.backtest.overnight import backtest_overnight, calculate_session_returns


def sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 110.0, 108.0],
            "close": [105.0, 108.0, 112.0],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )


def test_session_returns_use_previous_close_for_overnight_leg() -> None:
    returns = calculate_session_returns(sample_bars())

    assert math.isnan(returns.iloc[0]["overnight"])
    assert returns.iloc[1]["overnight"] == pytest.approx(110.0 / 105.0 - 1.0)
    assert returns.iloc[2]["overnight"] == pytest.approx(0.0)
    assert returns.iloc[1]["intraday"] == pytest.approx(108.0 / 110.0 - 1.0)
    assert returns.iloc[1]["close_to_close"] == pytest.approx(108.0 / 105.0 - 1.0)


def test_backtest_charges_cost_on_both_daily_executions() -> None:
    result = backtest_overnight(sample_bars(), cost_bps_per_side=10.0)
    gross = 110.0 / 105.0
    first_net_factor = gross * (1.0 - 0.001) ** 2
    ending_factor = first_net_factor * (1.0 - 0.001) ** 2

    assert result.daily_returns.iloc[0] == pytest.approx(first_net_factor - 1.0)
    assert result.trades == 2
    assert result.ending_value == pytest.approx(ending_factor)


@pytest.mark.parametrize("cost", [-1.0, 10_001.0])
def test_backtest_rejects_invalid_costs(cost: float) -> None:
    with pytest.raises(ValueError, match="cost_bps_per_side"):
        backtest_overnight(sample_bars(), cost_bps_per_side=cost)


def test_backtest_rejects_missing_or_nonpositive_prices() -> None:
    with pytest.raises(ValueError, match="open and close"):
        backtest_overnight(sample_bars().drop(columns="open"))

    bad = sample_bars()
    bad.loc[bad.index[1], "open"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        backtest_overnight(bad)
