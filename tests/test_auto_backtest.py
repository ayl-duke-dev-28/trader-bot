"""Behavioral tests for scored, bounded automatic backtest optimization."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.backtest.optimizer import (
    AutoBacktestOptimizer,
    OptimizationSettings,
    ParameterSpec,
    score_backtest,
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
