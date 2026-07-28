"""Tests for point-in-time macro-cycle scoring and exposure controls."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.macro import (
    build_initial_release_panel,
    fetch_fred_initial_releases,
    macro_cycle_history,
    macro_exposure_cap,
)


def _cycle_config(**overrides):
    values = {
        "enabled": True,
        "long_window_months": 60,
        "short_window_months": 6,
        "min_history_months": 24,
        "long_weight": 0.6,
        "short_weight": 0.4,
        "expansion_threshold": 0.20,
        "contraction_threshold": -0.20,
        "neutral_max_gross_exposure": 0.60,
        "contraction_max_gross_exposure": 0.30,
        "inflation_target_pct": 2.0,
    }
    values.update(overrides)
    return values


def _synthetic_macro() -> pd.DataFrame:
    dates = pd.date_range("2010-01-15", periods=144, freq="MS") + pd.Timedelta(days=14)
    split = 96
    growth = np.concatenate(
        [np.linspace(1.5, 4.0, split), np.linspace(4.0, -3.0, len(dates) - split)]
    )
    inflation = np.concatenate(
        [np.linspace(2.4, 2.0, split), np.linspace(2.0, 7.0, len(dates) - split)]
    )
    unemployment = np.concatenate(
        [np.linspace(7.0, 3.5, split), np.linspace(3.5, 8.0, len(dates) - split)]
    )
    interest_rate = np.concatenate(
        [np.linspace(2.0, 2.5, split), np.linspace(2.5, 7.0, len(dates) - split)]
    )
    return pd.DataFrame(
        {
            "growth": growth,
            "inflation": inflation,
            "unemployment": unemployment,
            "interest_rate": interest_rate,
        },
        index=dates,
    )


def test_macro_cycle_scores_long_and_short_expansion_then_contraction():
    cycles = macro_cycle_history(_synthetic_macro(), _cycle_config())

    assert cycles.iloc[90]["long_score"] > 0
    assert cycles.iloc[90]["short_score"] > 0
    assert cycles.iloc[90]["regime"] == "expansion"
    assert cycles.iloc[-1]["long_score"] < 0
    assert cycles.iloc[-1]["short_score"] < 0
    assert cycles.iloc[-1]["regime"] == "contraction"


def test_macro_exposure_uses_only_information_available_as_of_date():
    cycles = macro_cycle_history(_synthetic_macro(), _cycle_config())
    contraction_date = cycles.index[-1]

    assert macro_exposure_cap(
        cycles,
        cycles.index[0] - pd.Timedelta(days=1),
        normal_max_gross=0.80,
        config=_cycle_config(),
    ) == 0.80
    assert macro_exposure_cap(
        cycles,
        contraction_date,
        normal_max_gross=0.80,
        config=_cycle_config(),
    ) == 0.30


def test_macro_overlay_never_increases_existing_exposure_cap():
    cycles = macro_cycle_history(_synthetic_macro(), _cycle_config())
    expansion_date = cycles.index[90]

    assert macro_exposure_cap(
        cycles,
        expansion_date,
        normal_max_gross=0.20,
        config=_cycle_config(),
    ) == 0.20


def test_multiple_release_events_in_one_month_do_not_count_as_multiple_months():
    dates = pd.date_range("2020-01-02", periods=48, freq="7D")
    panel = pd.DataFrame(
        {
            "growth": 2.0,
            "inflation": 2.0,
            "unemployment": 4.0,
            "interest_rate": 2.0,
        },
        index=dates,
    )

    cycles = macro_cycle_history(
        panel,
        _cycle_config(min_history_months=24, long_window_months=60),
    )

    assert len(cycles) < 24
    assert set(cycles["regime"]) == {"unknown"}


def test_initial_release_panel_aligns_series_by_release_date_without_lookahead():
    releases = {
        "INDPRO": pd.DataFrame(
            {
                "observation_date": ["2020-01-01", "2021-01-01", "2021-02-01"],
                "available_date": ["2020-02-15", "2021-02-15", "2021-03-15"],
                "value": [100.0, 103.0, 104.0],
            }
        ),
        "CPIAUCSL": pd.DataFrame(
            {
                "observation_date": ["2020-01-01", "2021-01-01", "2021-02-01"],
                "available_date": ["2020-02-12", "2021-02-12", "2021-03-12"],
                "value": [250.0, 255.0, 257.5],
            }
        ),
        "UNRATE": pd.DataFrame(
            {
                "observation_date": ["2021-01-01", "2021-02-01"],
                "available_date": ["2021-02-05", "2021-03-05"],
                "value": [6.0, 5.8],
            }
        ),
        "FEDFUNDS": pd.DataFrame(
            {
                "observation_date": ["2021-01-01", "2021-02-01"],
                "available_date": ["2021-02-01", "2021-03-01"],
                "value": [0.09, 0.08],
            }
        ),
    }

    panel = build_initial_release_panel(releases)

    before_growth_release = panel.loc[pd.Timestamp("2021-02-12")]
    after_growth_release = panel.loc[pd.Timestamp("2021-02-15")]
    assert pd.isna(before_growth_release["growth"])
    assert np.isclose(after_growth_release["growth"], 3.0)
    assert np.isclose(after_growth_release["inflation"], 2.0)
    assert after_growth_release["unemployment"] == 6.0
    assert after_growth_release["interest_rate"] == 0.09


def test_fred_fetch_requests_initial_release_observations():
    class FakeResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "observations": [
                    {
                        "date": "2020-01-01",
                        "realtime_start": "2020-02-15",
                        "value": "100.5",
                    },
                    {
                        "date": "2020-02-01",
                        "realtime_start": "2020-03-15",
                        "value": ".",
                    },
                ]
            }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, *, params, timeout):
            self.calls.append((url, params, timeout))
            return FakeResponse()

    session = FakeSession()
    releases = fetch_fred_initial_releases("a" * 32, session=session)

    assert set(releases) == {"INDPRO", "CPIAUCSL", "UNRATE", "FEDFUNDS"}
    assert all(call[1]["output_type"] == 4 for call in session.calls)
    assert all(call[1]["file_type"] == "json" for call in session.calls)
    assert all(call[1]["realtime_start"] == "1776-07-04" for call in session.calls)
    assert all(call[1]["realtime_end"] == "9999-12-31" for call in session.calls)
    assert releases["INDPRO"].iloc[0]["available_date"] == pd.Timestamp("2020-02-15")
    assert len(releases["INDPRO"]) == 1
