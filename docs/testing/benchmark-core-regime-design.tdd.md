# Benchmark-core regime separation — TDD evidence

## Design defect

The benchmark-core target inferred market risk from the final gross-exposure
cap. A neutral macro regime reduced that cap from 80% to 60%, which was then
mistaken for a QQQ market risk-off signal and selected the zero-percent core
target.

Allocation state and exposure capacity are now evaluated separately:

- QQQ trend risk-on + macro expansion/unknown: 50% QQQ target.
- QQQ trend risk-on + macro neutral: 50% QQQ target within a 60% gross cap.
- Macro contraction: zero-percent QQQ target and a 30% gross cap.
- QQQ trend risk-off: zero-percent QQQ target regardless of the macro label.

The neutral target remains 50% because the execution path currently closes
positions as whole-position sell orders. Introducing a lower neutral target
would require a separate, tested partial-rebalance feature.

## RED

The focused tests failed before the production change:

```text
FAILED test_benchmark_core_target_distinguishes_market_and_macro_regimes
TypeError: unexpected keyword argument 'market_risk_on'

FAILED test_backtest_does_not_churn_risk_on_benchmark_core_on_neutral_score
assert 0 == 1
```

## GREEN

Focused verification:

```text
2 passed, 1 warning in 1.06s
```

Full regression suite:

```text
61 passed, 2 warnings in 17.74s
```

The warnings are pre-existing dependency notices for `websockets.legacy` and
loading the saved XGBoost pickle with a newer XGBoost version.
