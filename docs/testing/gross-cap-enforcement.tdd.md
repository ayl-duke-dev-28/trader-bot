# Gross-cap enforcement — TDD evidence

## User journeys

- As the paper/live operator, I want a lower regime cap to reduce existing
  exposure so the configured limit is an enforced portfolio constraint.
- As the operator, I want a cap reduction to sell only the required quantity so
  a trim does not accidentally liquidate the whole position.
- As a backtest user, I want historical simulations to use the same cap-trimming
  behavior as live sizing.
- As the operator, I want a pending sell to block another sell for the same
  symbol so hourly cycles do not duplicate trim orders.

## RED evidence

Before implementation, the focused live tests failed because `TradeIntent` had
no partial-sell representation and an over-cap account produced no sell intent:

```text
TypeError: TradeIntent.__init__() got an unexpected keyword argument
'sell_entire_position'
assert 0 == 1
```

The simulator transition test also demonstrated that an 80% portfolio remained
at 80% after the macro regime activated a 60% cap:

```text
assert (actual_gross <= 0.600001).all()
actual_gross: 0.8 on every neutral-regime row
```

The pending-order test initially showed that only open buys were queried:

```text
assert [call(side='buy')] == [call(side='buy'), call(side='sell')]
```

The environment denied creation of `.git/index.lock`, so the RED checkpoint
could not be committed. The tests remained in the working tree while the
production implementation was added.

## GREEN evidence

Focused cap, execution, and simulator tests:

```text
3 passed, 1 warning in 0.77s
```

Final regression suite after cap-boundary and pending-sell coverage:

```text
67 passed, 2 warnings in 17.33s
```

The warnings are existing dependency notices for `websockets.legacy` and the
older pickled XGBoost model.

## Guarantees

| Guarantee | Test | Type |
| --- | --- | --- |
| A neutral 60% cap trims a live 70% portfolio by exactly $10,000 | `test_active_gross_cap_creates_only_the_required_partial_sell` | integration |
| Partial execution converts target dollars into a bounded share quantity | `test_partial_sell_execution_uses_target_dollars_instead_of_closing_position` | unit |
| QQQ appreciation above its core target is trimmed before non-core holdings | `test_gross_cap_trims_core_growth_above_target_before_other_holdings` | unit |
| Already-planned full exits count toward the cap and are not duplicated | `test_gross_cap_accounts_for_full_exits_already_planned` | unit |
| An expansion-to-neutral backtest transition reduces actual gross exposure to 60% | `test_backtest_trims_existing_positions_when_macro_cap_falls` | integration |
| Live trims use a partial sell order, preserve position state, and skip a pending sell | `test_live_trade_cycle_routes_cap_trim_as_partial_sell_order` | integration |

## Coverage and known gaps

The full suite covered the changed simulator, manager, and trader modules at a
combined 78% (`simulator` 88%, `manager` 73%, `trader` 64%). The lower combined
number is driven by existing untested scheduling, macro-loader, and emergency
liquidation branches in `trader.py`; the new cap journeys above are directly
covered. Order acceptance is still not fill reconciliation: an accepted Alpaca
order can remain pending, so the open-sell guard prevents duplicate hourly
submissions but does not replace a future order-state ledger.
