# Stress harness TDD evidence

## Source and user journey

No plan file was supplied. The journey was derived from the request to stress
test the trading bot:

As a paper-trading operator, I want repeatable offline adverse scenarios so I
can verify that extreme markets, incomplete data, high costs, and economic
contraction do not corrupt accounting or bypass configured risk limits.

## RED and GREEN

The tests were written before the harness existed. The RED gate was:

```text
ImportError while importing tests/test_stress.py
ModuleNotFoundError: No module named 'src.backtest.stress'
1 error
```

After implementation, the focused GREEN run was:

```text
4 passed in 2.71s
```

The final repository validation was:

```text
37 passed, 2 warnings in 2.39s
```

The two warnings are pre-existing: the saved XGBoost pickle was produced by an
older XGBoost version, and `websockets.legacy` is deprecated.

## Test specification

| # | Guarantee | Test | Type | Result |
| --- | --- | --- | --- | --- |
| 1 | Synthetic OHLCV is deterministic, finite, positive, and internally consistent | `test_synthetic_history_is_deterministic_and_has_valid_ohlcv` | Unit | PASS |
| 2 | The suite includes baseline, crash, bear, volatility, missing-data, high-cost, and macro-contraction scenarios without mutating its source history | `test_stress_scenarios_cover_market_data_cost_and_macro_failures` | Unit | PASS |
| 3 | Every scenario exercises the live-path simulator while preserving finite positive equity, nonnegative cash, and configured caps | `test_stress_suite_runs_offline_scenarios_and_checks_safety_invariants` | Integration | PASS |
| 4 | Macro contraction limits actual gross exposure to 30% | `test_stress_suite_runs_offline_scenarios_and_checks_safety_invariants` | Integration | PASS |
| 5 | Extreme trading costs cannot improve final equity over the identical baseline price path | `test_stress_suite_runs_offline_scenarios_and_checks_safety_invariants` | Integration | PASS |
| 6 | Reports are emitted in Markdown, CSV, and JSON formats | `test_stress_report_writes_machine_and_human_readable_results` | Integration | PASS |

## Coverage

```text
Name                     Stmts   Miss  Cover
--------------------------------------------
src/backtest/stress.py     192     12    94%
```

Python bytecode compilation, CLI help loading, and `git diff --check` also
completed successfully. No checkpoint commits were created because the
workspace already contained related uncommitted user-requested changes; RED and
GREEN evidence is preserved here instead of mixing repository history.

## Retained stress run

Command:

```text
python3 scripts/stress_test.py --out-dir reports/stress_tests/2026-07-31
```

The current saved ML model was included. All software safety invariants passed,
but the overall verdict was `WARN` because the volatility-spike scenario reached
a `-58.39%` maximum drawdown. The flash crash produced a `-19.16%` maximum
drawdown, the high-cost scenario returned `-27.05%`, and macro contraction held
actual gross exposure below 30%.

These synthetic results measure controlled behavior and sensitivity. They do
not estimate the probability of a scenario or predict future returns.
