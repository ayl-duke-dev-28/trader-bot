"""Behavioral tests for scored, bounded automatic backtest optimization."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.backtest.optimizer import (
    AutoBacktestOptimizer,
    OptimizationSettings,
    ParameterSpec,
    attach_holdout,
    combine_window_scores,
    optimizer_settings_from_config,
    parameter_specs_from_config,
    score_backtest,
    split_backtest_period,
    write_optimization_report,
)
from src.config import Config


def _summary(
    *,
    cagr: float,
    sharpe: float,
    max_drawdown: float,
    win_rate: float,
    worst_day: float,
    trades: int,
) -> dict[str, float | int]:
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "closed_win_rate": win_rate,
        "worst_day_return": worst_day,
        "trades": trades,
    }


def test_score_is_bounded_and_explains_strengths_and_failures():
    strong = score_backtest(
        _summary(
            cagr=0.22,
            sharpe=1.8,
            max_drawdown=-0.09,
            win_rate=0.64,
            worst_day=-0.018,
            trades=120,
        )
    )
    weak = score_backtest(
        _summary(
            cagr=-0.12,
            sharpe=-0.4,
            max_drawdown=-0.52,
            win_rate=0.31,
            worst_day=-0.11,
            trades=80,
        )
    )

    assert 80 <= strong.score <= 100
    assert strong.strengths
    assert 1 <= weak.score <= 30
    assert weak.weaknesses
    assert any("drawdown" in note.lower() for note in weak.weaknesses)
    assert set(strong.components) == {
        "annual_return",
        "risk_adjusted_return",
        "drawdown_control",
        "trade_quality",
        "daily_stability",
    }


def test_score_penalizes_a_backtest_without_enough_trading_evidence():
    assessment = score_backtest(
        _summary(
            cagr=0.18,
            sharpe=1.6,
            max_drawdown=-0.08,
            win_rate=0.0,
            worst_day=-0.01,
            trades=0,
        )
    )

    assert assessment.score <= 25
    assert any("trades" in note.lower() for note in assessment.weaknesses)


def test_optimizer_retests_parameter_changes_and_keeps_best_config():
    base = Config(
        raw={"risk": {"entry_score_threshold": 0.70}},
        api_key="",
        api_secret="",
        is_live=False,
    )

    def evaluator(cfg: Config):
        threshold = cfg.get("risk", "entry_score_threshold")
        distance = abs(threshold - 0.50)
        return _summary(
            cagr=0.22 - distance,
            sharpe=1.8 - 4 * distance,
            max_drawdown=-0.10 - distance,
            win_rate=0.62 - distance,
            worst_day=-0.02 - distance / 2,
            trades=100,
        )

    optimizer = AutoBacktestOptimizer(
        base,
        evaluator,
        parameters=[
            ParameterSpec(
                path=("risk", "entry_score_threshold"),
                values=(0.70, 0.60, 0.50, 0.40),
            )
        ],
        settings=OptimizationSettings(
            target_score=90,
            max_iterations=5,
            max_evaluations=10,
            min_improvement=1,
        ),
    )

    result = optimizer.run()

    assert result.best_config.get("risk", "entry_score_threshold") == 0.50
    assert result.best_score.score >= 90
    assert result.stop_reason == "target_score_reached"
    assert len(result.runs) >= 3
    assert [run.score for run in result.runs if run.accepted] == sorted(
        run.score for run in result.runs if run.accepted
    )
    # Optimization works on deep copies and never mutates the live config object.
    assert base.get("risk", "entry_score_threshold") == 0.70


def test_optimizer_stops_when_no_candidate_improves():
    base = Config(raw={"risk": {"max_position_pct": 0.05}}, api_key="", api_secret="", is_live=False)

    result = AutoBacktestOptimizer(
        base,
        lambda _cfg: _summary(
            cagr=0.08,
            sharpe=0.8,
            max_drawdown=-0.20,
            win_rate=0.50,
            worst_day=-0.04,
            trades=50,
        ),
        parameters=[
            ParameterSpec(
                path=("risk", "max_position_pct"),
                values=(0.03, 0.05, 0.07),
            )
        ],
        settings=OptimizationSettings(max_iterations=5, max_evaluations=10),
    ).run()

    assert result.stop_reason == "no_improvement"
    assert result.best_config.get("risk", "max_position_pct") == 0.05


def test_report_contains_scores_notes_iterations_and_replayable_config(tmp_path: Path):
    base = Config(raw={"risk": {"entry_score_threshold": 0.60}}, api_key="", api_secret="", is_live=False)
    result = AutoBacktestOptimizer(
        base,
        lambda cfg: _summary(
            cagr=0.10 + (0.60 - cfg.get("risk", "entry_score_threshold")),
            sharpe=1.0,
            max_drawdown=-0.15,
            win_rate=0.55,
            worst_day=-0.03,
            trades=60,
        ),
        parameters=[ParameterSpec(("risk", "entry_score_threshold"), (0.60, 0.50))],
        settings=OptimizationSettings(max_iterations=2, max_evaluations=4),
    ).run()

    write_optimization_report(result, tmp_path)

    assert (tmp_path / "optimization.json").exists()
    markdown = (tmp_path / "summary.md").read_text()
    assert "Best score" in markdown
    assert "What worked" in markdown
    assert "What needs work" in markdown
    assert "Iteration history" in markdown
    saved = yaml.safe_load((tmp_path / "optimized_config.yaml").read_text())
    assert saved == result.best_config.raw


def test_window_score_penalizes_a_candidate_that_fails_validation():
    overfit = combine_window_scores(
        {
            "development": _summary(
                cagr=0.25,
                sharpe=2.1,
                max_drawdown=-0.08,
                win_rate=0.66,
                worst_day=-0.01,
                trades=100,
            ),
            "validation": _summary(
                cagr=-0.08,
                sharpe=-0.1,
                max_drawdown=-0.42,
                win_rate=0.35,
                worst_day=-0.08,
                trades=40,
            ),
        }
    )
    stable = combine_window_scores(
        {
            "development": _summary(
                cagr=0.14,
                sharpe=1.25,
                max_drawdown=-0.15,
                win_rate=0.57,
                worst_day=-0.025,
                trades=80,
            ),
            "validation": _summary(
                cagr=0.13,
                sharpe=1.20,
                max_drawdown=-0.16,
                win_rate=0.56,
                worst_day=-0.028,
                trades=35,
            ),
        }
    )

    assert stable.assessment.score > overfit.assessment.score
    assert overfit.window_scores["development"].score > overfit.window_scores["validation"].score
    assert any("generalization" in note.lower() for note in overfit.assessment.weaknesses)


def test_optimizer_accepts_multi_window_evaluations():
    base = Config(raw={"risk": {"entry_score_threshold": 0.60}}, api_key="", api_secret="", is_live=False)

    def evaluator(cfg: Config):
        threshold = cfg.get("risk", "entry_score_threshold")
        validation_cagr = 0.15 if threshold == 0.50 else -0.05
        return combine_window_scores(
            {
                "development": _summary(
                    cagr=0.16,
                    sharpe=1.3,
                    max_drawdown=-0.14,
                    win_rate=0.56,
                    worst_day=-0.025,
                    trades=80,
                ),
                "validation": _summary(
                    cagr=validation_cagr,
                    sharpe=1.25 if threshold == 0.50 else 0.1,
                    max_drawdown=-0.15 if threshold == 0.50 else -0.35,
                    win_rate=0.56 if threshold == 0.50 else 0.4,
                    worst_day=-0.025 if threshold == 0.50 else -0.07,
                    trades=40,
                ),
            }
        )

    result = AutoBacktestOptimizer(
        base,
        evaluator,
        parameters=[ParameterSpec(("risk", "entry_score_threshold"), (0.60, 0.50))],
        settings=OptimizationSettings(target_score=90, max_iterations=2, max_evaluations=3),
    ).run()

    assert result.best_config.get("risk", "entry_score_threshold") == 0.50
    assert "window_scores" in result.best_summary


def test_optimizer_options_and_search_space_load_from_config():
    cfg = Config(
        raw={
            "backtest": {
                "auto_optimize": {
                    "target_score": 91,
                    "max_iterations": 3,
                    "max_evaluations": 12,
                    "min_improvement": 2,
                    "parameters": [
                        {
                            "path": "risk.entry_score_threshold",
                            "values": [0.45, 0.55, 0.65],
                        }
                    ],
                }
            }
        },
        api_key="",
        api_secret="",
        is_live=False,
    )

    settings = optimizer_settings_from_config(cfg)
    parameters = parameter_specs_from_config(cfg)

    assert settings == OptimizationSettings(91, 3, 12, 2)
    assert parameters == (
        ParameterSpec(("risk", "entry_score_threshold"), (0.45, 0.55, 0.65)),
    )


def test_period_split_reserves_a_strictly_unseen_holdout():
    windows = split_backtest_period(
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2025-12-31"),
        development_fraction=0.60,
        validation_fraction=0.20,
    )

    assert set(windows) == {"development", "validation", "holdout"}
    assert windows["development"][0] == pd.Timestamp("2020-01-01")
    assert windows["development"][1] < windows["validation"][0]
    assert windows["validation"][1] < windows["holdout"][0]
    assert windows["holdout"][1] == pd.Timestamp("2025-12-31")


def test_report_separates_untouched_holdout_score(tmp_path: Path):
    base = Config(raw={"risk": {"entry_score_threshold": 0.55}}, api_key="", api_secret="", is_live=False)
    result = AutoBacktestOptimizer(
        base,
        lambda _cfg: _summary(
            cagr=0.12,
            sharpe=1.1,
            max_drawdown=-0.16,
            win_rate=0.55,
            worst_day=-0.03,
            trades=60,
        ),
        parameters=[ParameterSpec(("risk", "entry_score_threshold"), (0.55, 0.60))],
        settings=OptimizationSettings(max_iterations=1, max_evaluations=2),
    ).run()
    result = attach_holdout(
        result,
        _summary(
            cagr=0.09,
            sharpe=0.9,
            max_drawdown=-0.18,
            win_rate=0.52,
            worst_day=-0.04,
            trades=30,
        ),
    )

    write_optimization_report(result, tmp_path)

    assert result.holdout_score is not None
    assert "Holdout score" in (tmp_path / "summary.md").read_text()
    assert '"holdout_score"' in (tmp_path / "optimization.json").read_text()
