"""Point-in-time macroeconomic cycle data and scoring.

The backtester consumes a monthly panel indexed by the date each observation
became available. Required columns are:

``growth``
    Year-over-year industrial-production growth, percent.
``inflation``
    Year-over-year CPI inflation, percent.
``unemployment``
    U-3 unemployment rate, percent.
``interest_rate``
    Effective federal-funds rate, percent.

FRED's initial-release observations can be converted to this panel without
using revised values before their historical publication dates.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

MACRO_COLUMNS = ("growth", "inflation", "unemployment", "interest_rate")
FRED_SERIES = {
    "growth": "INDPRO",
    "inflation": "CPIAUCSL",
    "unemployment": "UNRATE",
    "interest_rate": "FEDFUNDS",
}
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _cfg(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _month_key(value: Any) -> pd.Period:
    return pd.Timestamp(value).to_period("M")


def fetch_fred_initial_releases(
    api_key: str,
    *,
    timeout_seconds: float = 60.0,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch initial-release observations for the four configured FRED series."""
    if not api_key:
        raise ValueError("FRED_API_KEY is required")

    client = session or requests.Session()
    releases: dict[str, pd.DataFrame] = {}
    for series_id in FRED_SERIES.values():
        response = client.get(
            FRED_OBSERVATIONS_URL,
            params={
                "api_key": api_key,
                "series_id": series_id,
                "file_type": "json",
                "output_type": 4,
                "sort_order": "asc",
                "realtime_start": "1776-07-04",
                "realtime_end": "9999-12-31",
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
        rows = []
        for observation in observations:
            raw_value = observation.get("value")
            if raw_value in (None, "."):
                continue
            rows.append(
                {
                    "observation_date": pd.Timestamp(observation["date"]),
                    "available_date": pd.Timestamp(observation["realtime_start"]),
                    "value": float(raw_value),
                }
            )
        releases[series_id] = pd.DataFrame(
            rows,
            columns=["observation_date", "available_date", "value"],
        )
    return releases


def build_initial_release_panel(
    releases: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build an as-of macro panel from per-series initial releases.

    Values change only on ``available_date``. Growth and inflation are computed
    from same-series observations twelve months apart, using only values that
    had already been released by that date.
    """
    required = set(FRED_SERIES.values())
    missing = sorted(required - set(releases))
    if missing:
        raise ValueError(f"missing FRED release series: {', '.join(missing)}")

    events: list[tuple[pd.Timestamp, str, pd.Period, float]] = []
    for series_id in required:
        frame = releases[series_id]
        expected = {"observation_date", "available_date", "value"}
        if not expected.issubset(frame.columns):
            raise ValueError(f"{series_id} releases must contain {sorted(expected)}")
        for row in frame.loc[:, ["observation_date", "available_date", "value"]].itertuples(index=False):
            if pd.isna(row.value):
                continue
            events.append(
                (
                    pd.Timestamp(row.available_date).normalize(),
                    series_id,
                    _month_key(row.observation_date),
                    float(row.value),
                )
            )

    events.sort(key=lambda item: (item[0], item[1], item[2]))
    values_by_series: dict[str, dict[pd.Period, float]] = {
        series_id: {} for series_id in required
    }
    latest_period: dict[str, pd.Period] = {}
    output: list[dict[str, Any]] = []

    event_index = 0
    while event_index < len(events):
        available_date = events[event_index][0]
        while event_index < len(events) and events[event_index][0] == available_date:
            _, series_id, observation_month, value = events[event_index]
            values_by_series[series_id][observation_month] = value
            if series_id not in latest_period or observation_month >= latest_period[series_id]:
                latest_period[series_id] = observation_month
            event_index += 1

        def latest_level(series_id: str) -> float:
            period = latest_period.get(series_id)
            if period is None:
                return np.nan
            return values_by_series[series_id].get(period, np.nan)

        def latest_yoy(series_id: str) -> float:
            period = latest_period.get(series_id)
            if period is None:
                return np.nan
            current = values_by_series[series_id].get(period)
            prior = values_by_series[series_id].get(period - 12)
            if current is None or prior is None or prior <= 0:
                return np.nan
            return (current / prior - 1.0) * 100.0

        output.append(
            {
                "available_date": available_date,
                "growth": latest_yoy(FRED_SERIES["growth"]),
                "inflation": latest_yoy(FRED_SERIES["inflation"]),
                "unemployment": latest_level(FRED_SERIES["unemployment"]),
                "interest_rate": latest_level(FRED_SERIES["interest_rate"]),
            }
        )

    if not output:
        return pd.DataFrame(columns=MACRO_COLUMNS, index=pd.DatetimeIndex([], name="available_date"))
    return (
        pd.DataFrame(output)
        .set_index("available_date")
        .sort_index()
        .loc[:, list(MACRO_COLUMNS)]
    )


def load_macro_panel(path: str | Path) -> pd.DataFrame:
    """Load a saved as-of panel with an ``available_date`` column."""
    frame = pd.read_csv(path, parse_dates=["available_date"])
    missing = sorted(set(MACRO_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"macro panel missing columns: {', '.join(missing)}")
    return (
        frame.set_index("available_date")
        .sort_index()
        .loc[:, list(MACRO_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
    )


def write_macro_panel(panel: pd.DataFrame, path: str | Path) -> None:
    """Persist an as-of panel as CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.rename_axis("available_date").reset_index().to_csv(destination, index=False)


def macro_cycle_history(
    panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calculate long-cycle, short-cycle, composite scores, and regimes."""
    missing = sorted(set(MACRO_COLUMNS) - set(panel.columns))
    if missing:
        raise ValueError(f"macro panel missing columns: {', '.join(missing)}")
    if panel.empty:
        return pd.DataFrame(
            columns=["long_score", "short_score", "composite_score", "regime"],
            index=pd.DatetimeIndex([], name="available_date"),
        )

    data = panel.loc[:, list(MACRO_COLUMNS)].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")].apply(pd.to_numeric, errors="coerce")
    # Several indicators release on different days in the same month. Collapse
    # them to one conservative month-end snapshot so all "*_months" settings
    # measure calendar months rather than release-event rows.
    data = data.resample("ME").last().ffill()

    long_window = max(12, int(_cfg(config, "long_window_months", 120)))
    short_window = max(1, int(_cfg(config, "short_window_months", 6)))
    min_history = max(short_window + 1, int(_cfg(config, "min_history_months", 36)))
    inflation_target = float(_cfg(config, "inflation_target_pct", 2.0))

    rolling_unemployment = data["unemployment"].rolling(
        long_window,
        min_periods=min_history,
    ).median()
    growth_level = np.tanh(data["growth"] / 3.0)
    inflation_stability = np.tanh(
        (1.0 - (data["inflation"] - inflation_target).abs()) / 1.5
    )
    unemployment_level = -np.tanh(
        (data["unemployment"] - rolling_unemployment) / 1.5
    )
    real_rate_pressure = -np.tanh(
        (data["interest_rate"] - data["inflation"]) / 3.0
    )
    long_score = pd.concat(
        [
            growth_level,
            inflation_stability,
            unemployment_level,
            real_rate_pressure,
        ],
        axis=1,
    ).mean(axis=1, skipna=False)

    changes = data.diff(short_window)
    short_score = pd.concat(
        [
            np.tanh(changes["growth"] / 2.0),
            -np.tanh(changes["inflation"] / 2.0),
            -np.tanh(changes["unemployment"] / 1.0),
            -np.tanh(changes["interest_rate"] / 2.0),
        ],
        axis=1,
    ).mean(axis=1, skipna=False)

    history_count = data.notna().all(axis=1).cumsum()
    ready = history_count >= min_history
    long_score = long_score.where(ready)
    short_score = short_score.where(ready)

    long_weight = float(_cfg(config, "long_weight", 0.6))
    short_weight = float(_cfg(config, "short_weight", 0.4))
    total_weight = max(1e-9, long_weight + short_weight)
    composite = (
        long_score * long_weight + short_score * short_weight
    ) / total_weight
    composite = composite.clip(-1.0, 1.0)

    expansion_threshold = float(_cfg(config, "expansion_threshold", 0.20))
    contraction_threshold = float(_cfg(config, "contraction_threshold", -0.20))
    regime = pd.Series("neutral", index=data.index, dtype="object")
    regime.loc[composite >= expansion_threshold] = "expansion"
    regime.loc[composite <= contraction_threshold] = "contraction"
    regime.loc[composite.isna()] = "unknown"

    return pd.DataFrame(
        {
            "long_score": long_score,
            "short_score": short_score,
            "composite_score": composite,
            "regime": regime,
        },
        index=data.index,
    )


def macro_cycle_at(cycles: pd.DataFrame, as_of: Any) -> pd.Series | None:
    """Return the latest cycle row available on or before ``as_of``."""
    if cycles is None or cycles.empty:
        return None
    cutoff = pd.Timestamp(as_of)
    available = cycles.loc[cycles.index <= cutoff]
    if available.empty:
        return None
    return available.iloc[-1]


def macro_exposure_cap(
    cycles: pd.DataFrame,
    as_of: Any,
    *,
    normal_max_gross: float,
    config: Mapping[str, Any],
) -> float:
    """Apply the macro regime cap without ever increasing another risk cap."""
    if not bool(_cfg(config, "enabled", True)):
        return float(normal_max_gross)
    current = macro_cycle_at(cycles, as_of)
    if current is None or current["regime"] in {"unknown", "expansion"}:
        return float(normal_max_gross)
    if current["regime"] == "contraction":
        configured_cap = float(_cfg(config, "contraction_max_gross_exposure", 0.30))
    else:
        configured_cap = float(_cfg(config, "neutral_max_gross_exposure", 0.60))
    return max(0.0, min(float(normal_max_gross), configured_cap))
