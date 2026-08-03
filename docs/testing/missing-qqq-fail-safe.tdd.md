# Missing QQQ benchmark fail-safe — TDD evidence

## User journeys

- As the paper/live operator, I want missing, insufficient, or lagging QQQ
  history to block new buys so the bot does not assume a risk-on market without
  its configured benchmark.
- As the operator, I want risk-reducing exits to remain available during that
  data outage.
- As the operator, I do not want missing QQQ data alone to trigger a forced QQQ
  liquidation because unavailable data is not evidence of a downtrend.
- As a backtest user, I want the simulator to apply the same fail-safe behavior.

## RED evidence

The initial focused run produced two failures and one passing exit test:

```text
FAILED test_live_risk_manager_freezes_buys_when_qqq_benchmark_is_missing
Left contains TradeIntent(symbol='AAPL', side='buy', target_dollars=5000.0, ...)

FAILED test_backtest_freezes_buys_when_qqq_benchmark_is_missing
assert 1 == 0

2 failed, 1 passed, 1 warning in 1.19s
```

This proved that missing QQQ data was being treated as risk-on in both paths.
The RED checkpoint is commit `19eefdc`.

## GREEN evidence

Focused live and backtest verification:

```text
4 passed, 1 warning in 0.77s
```

Full suite with coverage:

```text
71 passed, 2 warnings in 19.89s
src/backtest/simulator.py  88%
src/risk/manager.py        76%
TOTAL                      83%
```

The warnings are existing dependency notices for `websockets.legacy` and the
older pickled XGBoost model.

## Guarantees

| Guarantee | Test | Type |
| --- | --- | --- |
| Missing QQQ blocks an otherwise eligible live AAPL buy without forcing an existing QQQ core sale | `test_live_risk_manager_freezes_buys_when_qqq_benchmark_is_missing` | integration |
| Negative-score exits still pass through while QQQ is unavailable | `test_live_risk_manager_still_allows_exits_when_qqq_benchmark_is_missing` | integration |
| Fewer than the configured SMA bars and a QQQ series lagging other market data are unavailable states | `test_market_regime_state_rejects_short_or_lagging_qqq_history` | unit |
| The historical simulator opens no positions when its configured QQQ benchmark is absent | `test_backtest_freezes_buys_when_qqq_benchmark_is_missing` | integration |

## Design boundary

The regime state is tri-state: risk-on, risk-off, or unavailable. Unavailable
freezes entries but does not choose the risk-off QQQ target. Macro contraction
remains independently authoritative, and risk-reducing score exits and active
gross-cap trims remain permitted. The simulator records daily benchmark-data
availability and reports the number of unavailable days.
