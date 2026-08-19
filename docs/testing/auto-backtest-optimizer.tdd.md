# Automatic backtest optimizer TDD evidence

## Source and user journeys

No plan file was supplied. These journeys were derived from the request:

- As a trader, I want every backtest scored from 1–100 with strengths and
  weaknesses, so I can compare results consistently.
- As a trader, I want the bot to adjust only approved strategy settings and
  re-run the backtest until it reaches a target or a bounded stopping rule.
- As a trader, I want validation and untouched holdout results, so a tuning win
  is less likely to be mistaken for future performance.
- As an operator, I want a replayable config and recoverable apply operation,
  so optimization cannot silently or irreversibly alter live settings.

## RED and GREEN evidence

| Behavior | RED evidence | GREEN evidence | Guarantee |
| --- | --- | --- | --- |
| Score, notes, loop, and artifacts | `python3 -m pytest tests/test_auto_backtest.py -q` failed with `ModuleNotFoundError: src.backtest.optimizer` | Five tests passed | Scores remain in 1–100, sparse trading is penalized, improved candidates are re-tested and retained, and reports are replayable |
| Validation-aware objective | The targeted suite failed importing missing `combine_window_scores` | Eight tests passed | Validation receives greater weight and a material development/validation gap is penalized |
| Untouched holdout | The targeted suite failed importing missing `attach_holdout` | Ten tests passed | The final chronological window is excluded from tuning and reported separately |
| Safe apply | The targeted suite failed with `ModuleNotFoundError: scripts.auto_backtest` | Eleven tests passed | Applying a winning config first saves the active config beside it |
| CLI orchestration | Added offline integration coverage with all network and backtest execution mocked | Twelve tests passed | One baseline candidate causes development, validation, and final holdout runs and emits the expected artifacts/output |
| Immediate score threshold | The targeted test showed evaluations continuing after a candidate crossed 85 | Thirteen tests passed | The first candidate at or above the configured target is accepted and no later candidate is run |

All RED/GREEN checkpoints are separate commits on `main`; none were squashed.

## Test specification

| # | What is guaranteed | Test target | Type | Result |
| ---: | --- | --- | --- | --- |
| 1 | Strong and weak metrics produce bounded, explainable scores | `test_score_is_bounded_and_explains_strengths_and_failures` | Unit | PASS |
| 2 | Zero/sparse trading cannot game the score | `test_score_penalizes_a_backtest_without_enough_trading_evidence` | Unit | PASS |
| 3 | Candidate configs are re-tested, only improvements are accepted, and the caller config is unchanged | `test_optimizer_retests_parameter_changes_and_keeps_best_config` | Unit | PASS |
| 4 | Search stops when candidates do not improve | `test_optimizer_stops_when_no_candidate_improves` | Unit | PASS |
| 5 | JSON, Markdown, and proposed YAML artifacts contain the audit trail | `test_report_contains_scores_notes_iterations_and_replayable_config` | Integration | PASS |
| 6 | Validation collapse loses to stable performance | `test_window_score_penalizes_a_candidate_that_fails_validation` | Unit | PASS |
| 7 | The optimizer consumes multi-window objectives | `test_optimizer_accepts_multi_window_evaluations` | Integration | PASS |
| 8 | Settings and allow-listed parameter values load from config | `test_optimizer_options_and_search_space_load_from_config` | Unit | PASS |
| 9 | Chronological windows do not overlap and reserve a final holdout | `test_period_split_reserves_a_strictly_unseen_holdout` | Unit | PASS |
| 10 | Holdout score is distinct in JSON and Markdown | `test_report_separates_untouched_holdout_score` | Integration | PASS |
| 11 | Explicit apply preserves the prior active config | `test_explicit_apply_keeps_a_recoverable_config_backup` | Integration | PASS |
| 12 | CLI orchestration executes development, validation, and holdout offline | `test_cli_runs_tuning_validation_and_holdout_without_network` | Integration | PASS |
| 13 | Reaching 85 stops candidate evaluation immediately | `test_optimizer_stops_evaluating_immediately_at_target_score` | Unit | PASS |

## Validation and coverage

Commands run:

```text
PYTHONPATH=.venv/lib/python3.14/site-packages python3 -m pytest \
  tests/test_auto_backtest.py --cov=src.backtest.optimizer \
  --cov=scripts.auto_backtest --cov-report=term-missing -q
13 passed
src/backtest/optimizer.py: 86%
scripts/auto_backtest.py: 81%
combined feature coverage: 85%

PYTHONPATH=.venv/lib/python3.14/site-packages python3 -m pytest -q
103 passed, 7 warnings
```

`python3 -m compileall -q src scripts tests`, CLI `--help`, and
`git diff --check` also passed. The warnings are existing websockets and
`datetime.utcnow()` deprecations plus the existing cross-version XGBoost model
serialization warning.

## Known gaps and safety boundary

No network market-data run was performed as part of the test suite. The offline
CLI integration test mocks data retrieval and backtest execution; the existing
backtest suite covers the underlying simulator.

The optimizer does not rewrite Python, expand its own search space, enable live
mode, or promise a globally optimal strategy. It searches the finite approved
grid, uses configured evaluation limits, writes a proposal by default, and
requires explicit `--apply` before changing the active config. Historical and
holdout results remain research evidence, not forecasts.
