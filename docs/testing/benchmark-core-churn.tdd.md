# Benchmark-core QQQ churn fix — TDD evidence

## Source and user journey

No plan file was supplied. The journey was derived from the observed paper
orders: as a trader using a risk-on QQQ core sleeve, I want a negative ensemble
score to leave that sleeve in place so the bot does not sell and repurchase the
same core position every hour. Risk-off and stop-driven exits must remain.

## RED and GREEN report

| Behavior | RED evidence | GREEN evidence | Guarantee |
|---|---|---|---|
| Live sizing retains an active QQQ core despite a negative score | Focused pytest run failed because `size_orders` returned `TradeIntent(symbol='QQQ', side='sell', ...)` | The same focused run passed | Generic score exits cannot liquidate a benchmark core whose configured target is positive. |
| Historical simulation matches live behavior | Focused pytest run failed with `15 == 1` benchmark buys | The same focused run passed with one initial buy and zero score-driven sells | The simulator does not hide or exaggerate the live fix through daily sell/rebuy churn. |
| Risk-off liquidation remains intact | N/A; this was existing intended behavior | `test_benchmark_core_still_exits_when_regime_target_is_zero` passed | A zero core target still creates a full QQQ sell intent. |

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | Negative QQQ score does not sell a correctly sized risk-on core | `test_risk_on_benchmark_core_ignores_negative_score_exit` | Unit/integration | PASS |
| 2 | A zero risk-off target still liquidates QQQ | `test_benchmark_core_still_exits_when_regime_target_is_zero` | Unit | PASS |
| 3 | A neutral-scored QQQ core is bought once and held in the live-path backtest | `test_backtest_does_not_churn_risk_on_benchmark_core_on_neutral_score` | Integration | PASS |

## Validation and coverage

- RED command: `PYTHONPATH=.venv/lib/python3.14/site-packages python3 -m pytest tests/test_smoke.py::test_risk_on_benchmark_core_ignores_negative_score_exit tests/test_smoke.py::test_backtest_does_not_churn_risk_on_benchmark_core_on_neutral_score -q`
- GREEN focused result: `3 passed` after adding the risk-off boundary test.
- Full regression result: `60 passed`, with the existing websockets deprecation
  and serialized-XGBoost version warnings.
- Full-suite coverage: `src/backtest/simulator.py` 87%,
  `src/risk/manager.py` 71%, combined changed-module coverage 81%.

No broker orders or external writes were made during verification.

## Checkpoints

- RED: `e85df60 test: reproduce benchmark core QQQ churn`
- GREEN: production and evidence commit recorded after full validation.
