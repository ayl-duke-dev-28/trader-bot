"""Offline stress-harness tests. No market data or broker calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.stress import (
    build_stress_scenarios,
    make_synthetic_history,
    run_stress_suite,
    write_stress_report,
)
from src.config import load_config


def test_synthetic_history_is_deterministic_and_has_valid_ohlcv():
    symbols = ("QQQ", "AAPL", "NVDA")
    first = make_synthetic_history(symbols=symbols, periods=280, seed=17)
    second = make_synthetic_history(symbols=symbols, periods=280, seed=17)

    assert set(first) == set(symbols)
    for symbol in symbols:
        assert first[symbol].equals(second[symbol])
        frame = first[symbol]
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all().all()
        assert (frame[["open", "high", "low", "close", "volume"]] > 0).all().all()
        assert (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        assert (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()


def test_stress_scenarios_cover_market_data_cost_and_macro_failures():
    history = make_synthetic_history(periods=300, seed=5)
    original_qqq = history["QQQ"].copy(deep=True)
    scenarios = build_stress_scenarios(history, simulation_bars=60)

    assert set(scenarios) == {
        "baseline",
        "flash_crash",
        "prolonged_bear",
        "volatility_spike",
        "missing_data",
        "high_cost",
        "macro_contraction",
    }
    assert history["QQQ"].equals(original_qqq)
    assert scenarios["flash_crash"].history["QQQ"]["close"].iloc[-30] < original_qqq["close"].iloc[-30]
    assert scenarios["high_cost"].cost_bps > scenarios["baseline"].cost_bps
    assert scenarios["macro_contraction"].macro_cycles is not None
    assert scenarios["macro_contraction"].macro_config["enabled"] is True
    assert scenarios["missing_data"].history["AAPL"]["close"].isna().any()


def test_stress_suite_runs_offline_scenarios_and_checks_safety_invariants():
    cfg = load_config(require_secrets=False)
    suite = run_stress_suite(
        cfg,
        periods=520,
        simulation_bars=252,
        seed=11,
        include_saved_model=False,
    )

    assert len(suite.results) == 7
    assert suite.total_runtime_seconds >= 0
    assert all(result.status in {"PASS", "WARN", "FAIL"} for result in suite.results)
    assert all(np.isfinite(result.final_equity) for result in suite.results)
    assert all(result.final_equity > 0 for result in suite.results)
    assert all(result.min_cash >= -0.01 for result in suite.results)
    assert all(result.invariants_passed for result in suite.results)
    assert suite.by_name("macro_contraction").max_gross_exposure <= 0.30
    assert suite.by_name("high_cost").final_equity <= suite.by_name("baseline").final_equity
    assert suite.by_name("volatility_spike").var_blocked_buys > 0


def test_stress_report_writes_machine_and_human_readable_results():
    cfg = load_config(require_secrets=False)
    suite = run_stress_suite(
        cfg,
        periods=260,
        simulation_bars=30,
        seed=3,
        include_saved_model=False,
        scenario_names=("baseline", "flash_crash"),
    )

    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        write_stress_report(suite, out_dir)

        assert (out_dir / "summary.md").exists()
        assert (out_dir / "results.csv").exists()
        payload = json.loads((out_dir / "results.json").read_text())
        assert payload["seed"] == 3
        assert [row["scenario"] for row in payload["results"]] == ["baseline", "flash_crash"]
        assert "Safety verdict" in (out_dir / "summary.md").read_text()
