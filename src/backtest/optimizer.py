"""Transparent scoring and bounded configuration search for live-path backtests.

The optimizer deliberately changes only allow-listed configuration values.  It
never edits strategy source code, never updates the caller's ``Config`` object,
and always stops at a configured target, iteration limit, evaluation limit, or
when no candidate improves the score.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.config import Config


Summary = Mapping[str, Any]
Evaluator = Callable[[Config], Summary | Any]


@dataclass(frozen=True)
class BacktestScore:
    """A reproducible 1-100 assessment with human-readable diagnostics."""

    score: int
    components: dict[str, float]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "components": self.components,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
        }


@dataclass(frozen=True)
class ParameterSpec:
    """An allow-listed config leaf and the finite values it may take."""

    path: tuple[str, ...]
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("parameter path cannot be empty")
        if not self.values:
            raise ValueError(f"parameter {'.'.join(self.path)} has no candidate values")

    @property
    def name(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class OptimizationSettings:
    target_score: int = 85
    max_iterations: int = 6
    max_evaluations: int = 40
    min_improvement: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.target_score <= 100:
            raise ValueError("target_score must be between 1 and 100")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if self.min_improvement < 0:
            raise ValueError("min_improvement cannot be negative")


@dataclass(frozen=True)
class OptimizationRun:
    evaluation: int
    iteration: int
    changes: dict[str, Any]
    score: int
    components: dict[str, float]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    summary: dict[str, Any]
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["strengths"] = list(self.strengths)
        value["weaknesses"] = list(self.weaknesses)
        return _json_safe(value)


@dataclass(frozen=True)
class OptimizationResult:
    base_config: Config
    best_config: Config
    best_score: BacktestScore
    best_summary: dict[str, Any]
    runs: tuple[OptimizationRun, ...]
    stop_reason: str
    settings: OptimizationSettings


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("score range must have a positive width")
    return round(max(0.0, min(100.0, 100.0 * (value - low) / (high - low))), 2)


def score_backtest(summary: Summary) -> BacktestScore:
    """Score a backtest from 1-100 using explicit, stable metric bands.

    The score rewards return, risk-adjusted return, drawdown control, realized
    trade quality, and daily loss containment.  Sparse runs are capped because
    attractive metrics based on only a handful of trades are weak evidence.
    """
    cagr = _finite_float(summary.get("cagr"))
    sharpe = _finite_float(summary.get("sharpe"))
    drawdown = abs(_finite_float(summary.get("max_drawdown")))
    win_rate = _finite_float(summary.get("closed_win_rate"))
    worst_day = abs(_finite_float(summary.get("worst_day_return")))
    trades = max(0, int(_finite_float(summary.get("trades"))))

    components = {
        "annual_return": _scale(cagr, -0.10, 0.20),
        "risk_adjusted_return": _scale(sharpe, -0.50, 2.00),
        "drawdown_control": _scale(-drawdown, -0.50, -0.10),
        "trade_quality": _scale(win_rate, 0.30, 0.65),
        "daily_stability": _scale(-worst_day, -0.10, -0.015),
    }
    weights = {
        "annual_return": 0.25,
        "risk_adjusted_return": 0.30,
        "drawdown_control": 0.25,
        "trade_quality": 0.10,
        "daily_stability": 0.10,
    }
    raw_score = sum(components[name] * weight for name, weight in weights.items())

    weaknesses: list[str] = []
    strengths: list[str] = []
    descriptions = {
        "annual_return": (f"CAGR was {cagr:.1%}", f"CAGR was only {cagr:.1%}"),
        "risk_adjusted_return": (
            f"Sharpe ratio was {sharpe:.2f}",
            f"Sharpe ratio was weak at {sharpe:.2f}",
        ),
        "drawdown_control": (
            f"Maximum drawdown was contained at {drawdown:.1%}",
            f"Maximum drawdown was high at {drawdown:.1%}",
        ),
        "trade_quality": (
            f"Closed-trade win rate was {win_rate:.1%}",
            f"Closed-trade win rate was low at {win_rate:.1%}",
        ),
        "daily_stability": (
            f"Worst day was limited to {worst_day:.1%}",
            f"Worst day loss was high at {worst_day:.1%}",
        ),
    }
    for name, component_score in components.items():
        good, bad = descriptions[name]
        if component_score >= 70:
            strengths.append(good)
        elif component_score <= 45:
            weaknesses.append(bad)

    if trades == 0:
        raw_score = min(raw_score, 25.0)
        weaknesses.append("No trades were executed, so the performance has no trading evidence")
    elif trades < 10:
        raw_score *= 0.55
        weaknesses.append(f"Only {trades} trades were executed; results are too sparse to trust")
    elif trades < 30:
        raw_score *= 0.80
        weaknesses.append(f"Only {trades} trades were executed; confidence is limited")
    else:
        strengths.append(f"The result includes {trades} executed trades")

    return BacktestScore(
        score=max(1, min(100, int(round(raw_score)))),
        components=components,
        strengths=tuple(strengths),
        weaknesses=tuple(weaknesses),
    )


def _config_copy(cfg: Config) -> Config:
    return Config(
        raw=deepcopy(cfg.raw),
        api_key=cfg.api_key,
        api_secret=cfg.api_secret,
        is_live=cfg.is_live,
    )


def _read_path(raw: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = raw
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _write_path(raw: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = raw
    for key in path[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {'.'.join(path)} through non-mapping key {key}")
        node = child
    node[path[-1]] = deepcopy(value)


def _signature(cfg: Config, parameters: Sequence[ParameterSpec]) -> str:
    values = [(spec.name, _read_path(cfg.raw, spec.path)) for spec in parameters]
    return json.dumps(values, sort_keys=True, default=str)


def _extract_summary(result: Summary | Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    summary = getattr(result, "summary", None)
    if isinstance(summary, Mapping):
        return dict(summary)
    raise TypeError("evaluator must return a summary mapping or an object with a summary mapping")


class AutoBacktestOptimizer:
    """Coordinate search that backtests every proposed configuration."""

    def __init__(
        self,
        base_config: Config,
        evaluator: Evaluator,
        parameters: Sequence[ParameterSpec],
        settings: OptimizationSettings | None = None,
    ) -> None:
        if not parameters:
            raise ValueError("at least one optimization parameter is required")
        self.base_config = _config_copy(base_config)
        self.evaluator = evaluator
        self.parameters = tuple(parameters)
        self.settings = settings or OptimizationSettings()

    def _evaluate(
        self,
        cfg: Config,
        *,
        evaluation: int,
        iteration: int,
        changes: dict[str, Any],
        accepted: bool,
    ) -> tuple[BacktestScore, dict[str, Any], OptimizationRun]:
        summary = _extract_summary(self.evaluator(cfg))
        assessment = score_backtest(summary)
        run = OptimizationRun(
            evaluation=evaluation,
            iteration=iteration,
            changes=deepcopy(changes),
            score=assessment.score,
            components=assessment.components,
            strengths=assessment.strengths,
            weaknesses=assessment.weaknesses,
            summary=_json_safe(summary),
            accepted=accepted,
        )
        return assessment, summary, run

    def run(self) -> OptimizationResult:
        current = _config_copy(self.base_config)
        best_score, best_summary, baseline = self._evaluate(
            current,
            evaluation=1,
            iteration=0,
            changes={},
            accepted=True,
        )
        runs: list[OptimizationRun] = [baseline]
        seen = {_signature(current, self.parameters)}
        evaluation_count = 1

        if best_score.score >= self.settings.target_score:
            stop_reason = "target_score_reached"
        else:
            stop_reason = "max_iterations_reached"
            for iteration in range(1, self.settings.max_iterations + 1):
                candidates: list[tuple[BacktestScore, dict[str, Any], Config, int]] = []
                limit_hit = False
                for spec in self.parameters:
                    current_value = _read_path(current.raw, spec.path)
                    for value in spec.values:
                        if value == current_value:
                            continue
                        candidate = _config_copy(current)
                        _write_path(candidate.raw, spec.path, value)
                        signature = _signature(candidate, self.parameters)
                        if signature in seen:
                            continue
                        if evaluation_count >= self.settings.max_evaluations:
                            limit_hit = True
                            break
                        seen.add(signature)
                        evaluation_count += 1
                        assessment, summary, run = self._evaluate(
                            candidate,
                            evaluation=evaluation_count,
                            iteration=iteration,
                            changes={spec.name: value},
                            accepted=False,
                        )
                        runs.append(run)
                        candidates.append((assessment, summary, candidate, len(runs) - 1))
                    if limit_hit:
                        break

                if not candidates:
                    stop_reason = "max_evaluations_reached" if limit_hit else "no_improvement"
                    break

                candidate_score, candidate_summary, candidate_cfg, run_index = max(
                    candidates, key=lambda item: item[0].score
                )
                improvement = candidate_score.score - best_score.score
                if improvement < self.settings.min_improvement:
                    stop_reason = "no_improvement"
                    break

                current = candidate_cfg
                best_score = candidate_score
                best_summary = candidate_summary
                runs[run_index] = OptimizationRun(**{**asdict(runs[run_index]), "accepted": True})

                if best_score.score >= self.settings.target_score:
                    stop_reason = "target_score_reached"
                    break
                if limit_hit or evaluation_count >= self.settings.max_evaluations:
                    stop_reason = "max_evaluations_reached"
                    break

        return OptimizationResult(
            base_config=_config_copy(self.base_config),
            best_config=_config_copy(current),
            best_score=best_score,
            best_summary=_json_safe(best_summary),
            runs=tuple(runs),
            stop_reason=stop_reason,
            settings=self.settings,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def write_optimization_report(result: OptimizationResult, out_dir: Path) -> None:
    """Write a replayable config plus machine- and human-readable audit logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_score": result.best_score.to_dict(),
        "best_summary": result.best_summary,
        "stop_reason": result.stop_reason,
        "settings": asdict(result.settings),
        "runs": [run.to_dict() for run in result.runs],
    }
    (out_dir / "optimization.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "optimized_config.yaml").write_text(
        yaml.safe_dump(result.best_config.raw, sort_keys=False)
    )

    strengths = result.best_score.strengths or ("No component crossed the strong threshold.",)
    weaknesses = result.best_score.weaknesses or ("No major weakness crossed the alert threshold.",)
    lines = [
        "# Automatic backtest optimization",
        "",
        f"**Best score:** {result.best_score.score}/100",
        f"**Stop reason:** `{result.stop_reason}`",
        f"**Backtests executed:** {len(result.runs)}",
        "",
        "## What worked",
        "",
        *[f"- {note}" for note in strengths],
        "",
        "## What needs work",
        "",
        *[f"- {note}" for note in weaknesses],
        "",
        "## Score components",
        "",
        "| Component | Score |",
        "| --- | ---: |",
        *[
            f"| {name.replace('_', ' ').title()} | {score:.1f}/100 |"
            for name, score in result.best_score.components.items()
        ],
        "",
        "## Iteration history",
        "",
        "| Evaluation | Round | Score | Accepted | Change |",
        "| ---: | ---: | ---: | :---: | --- |",
    ]
    for run in result.runs:
        change = ", ".join(f"`{key}={value}`" for key, value in run.changes.items()) or "baseline"
        lines.append(
            f"| {run.evaluation} | {run.iteration} | {run.score} | "
            f"{'yes' if run.accepted else 'no'} | {change} |"
        )
    lines.extend(
        [
            "",
            "The score is a comparative research aid, not a guarantee of future returns. ",
            "Use the optimized config in a genuinely unseen period and paper trading before live deployment.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines))
