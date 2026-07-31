# Portfolio VaR gate TDD evidence

## Source and user journey

No plan file was supplied. The journey was derived from the request to
implement VaR in the trading bot:

As a paper-trading operator, I want projected portfolio market risk checked
before each buy so that the bot does not add exposure above explicit one-day
historical VaR and expected-shortfall limits while always preserving its ability
to sell.

## Design

- Method: historical simulation using aligned daily close-to-close returns.
- Horizon: one trading day.
- Current configuration: 99% confidence, 252-day lookback, minimum 60 aligned
  observations, 3% of equity VaR limit, and 4% expected-shortfall limit.
- Rollout: enabled in paper configuration and replayed by the live-path
  backtester.
- Failure behavior: sells always pass; projected buys fail closed if data is
  insufficient or either risk limit is breached.

## RED and GREEN

Initial RED:

```text
ModuleNotFoundError: No module named 'src.risk.var'
```

After the estimator existed, the live/backtest integration tests remained RED:

```text
test_live_risk_manager_applies_var_gate_to_benchmark_core_buy: expected [] but received a QQQ buy
test_simulator_records_var_diagnostics_and_blocks_oversized_core_buy: KeyError: 'var_enabled'
```

Final focused GREEN:

```text
7 passed
```

Final repository validation:

```text
44 passed, 2 warnings in 7.34s
```

The warnings are pre-existing XGBoost serialization and `websockets.legacy`
deprecation warnings.

## Test specification

| # | Guarantee | Test | Type | Result |
| --- | --- | --- | --- | --- |
| 1 | Historical VaR equals the configured portfolio loss quantile | `test_historical_var_and_expected_shortfall_match_portfolio_loss_distribution` | Unit | PASS |
| 2 | Expected shortfall averages losses at and beyond the VaR cutoff | same test | Unit | PASS |
| 3 | Cross-symbol return alignment preserves diversification | `test_portfolio_var_preserves_cross_symbol_diversification` | Unit | PASS |
| 4 | Invalid equity and insufficient aligned observations return no estimate | `test_var_returns_none_for_invalid_equity_or_insufficient_aligned_history` | Unit | PASS |
| 5 | Sells and safe buys pass while an excessive projected buy is blocked | `test_var_gate_allows_sells_and_safe_buys_but_blocks_excess_risk` | Unit | PASS |
| 6 | An enabled gate fails closed on missing history and a disabled gate is a no-op | `test_var_gate_fails_closed_when_history_is_missing_and_can_be_disabled` | Unit | PASS |
| 7 | Live benchmark-core sizing uses the same gate | `test_live_risk_manager_applies_var_gate_to_benchmark_core_buy` | Integration | PASS |
| 8 | Backtests record daily VaR/ES and blocked-buy diagnostics | `test_simulator_records_var_diagnostics_and_blocks_oversized_core_buy` | Integration | PASS |
| 9 | Stress reports expose the number of VaR-blocked buys | `test_stress_suite_runs_offline_scenarios_and_checks_safety_invariants` | Integration | PASS |

## Coverage and stress comparison

```text
Name              Stmts   Miss  Cover
-------------------------------------
src/risk/var.py      90      1    99%
```

The retained saved-model stress comparison used identical seed, history,
scenarios, and configuration except for the new gate:

| Volatility-spike result | Before VaR | VaR enabled |
| --- | ---: | ---: |
| Return | -57.88% | -31.10% |
| Maximum drawdown | -58.39% | -31.93% |
| Trades | 120 | 71 |
| VaR-blocked buys | unavailable | 64 |

The instantaneous flash-crash result did not improve because historical VaR
cannot anticipate an unprecedented return before it enters the lookback window.
This is why expected shortfall, gap controls, drawdown controls, and stress tests
remain separate safeguards.

No checkpoint commits were created because the working tree already contained
related user-requested changes. RED/GREEN evidence is preserved here without
mixing repository history.
