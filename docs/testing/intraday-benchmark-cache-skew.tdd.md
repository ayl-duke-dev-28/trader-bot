# Intraday benchmark cache skew — TDD evidence

## Source and user journey

The journey was derived during this TDD run; no external plan file was used.

- As the paper/live operator, I want the bot to accept QQQ's latest completed
  session while other symbols contain a current-session partial daily bar, so
  an ordinary cache refresh skew does not suppress valid buys.
- As the operator, I still want missing, short, or genuinely stale QQQ history
  to fail closed and block new risk.

## RED evidence

The new reproducer was executed before production code changed:

```text
FAILED test_market_regime_state_accepts_previous_close_during_current_session
TypeError: RiskManager._market_regime_state() got an unexpected keyword argument 'as_of'
1 failed, 1 warning in 1.63s
```

This proved that the regime check could not distinguish a current-session
partial-bar skew from genuinely lagging benchmark data. RED checkpoint:
`a8afd1d`.

## GREEN evidence

The focused safeguard run after the minimal fix:

```text
4 passed, 1 warning in 1.13s
```

The full suite and affected-module coverage run:

```text
84 passed, 7 warnings in 18.13s
src/data/market_data.py  76%
src/risk/manager.py      86%
TOTAL                    85%
```

GREEN checkpoint: `fc8824c`.

## Guarantees

| # | What is guaranteed | Test | Type | Result |
| --- | --- | --- | --- | --- |
| 1 | Previous-session QQQ data remains usable when another symbol has the current session's partial bar | `test_market_regime_state_accepts_previous_close_during_current_session` | Unit | PASS |
| 2 | QQQ that is two business sessions stale remains unavailable | `test_market_regime_state_accepts_previous_close_during_current_session` | Boundary | PASS |
| 3 | Missing or insufficient QQQ continues to block live buys while allowing risk-reducing exits | `test_live_risk_manager_freezes_buys_when_qqq_benchmark_is_missing`, `test_live_risk_manager_still_allows_exits_when_qqq_benchmark_is_missing` | Integration | PASS |
| 4 | Historical one-bar lag outside the current intraday session remains unavailable | `test_market_regime_state_rejects_short_or_lagging_qqq_history` | Unit | PASS |

## Coverage and known gaps

The affected modules have 85% combined coverage. Repository-wide coverage is
78% because unrelated broker, earnings, politician-tracking, and universe
modules have pre-existing gaps. The seven warnings are pre-existing dependency,
model-serialization, and naive-UTC deprecation warnings.
