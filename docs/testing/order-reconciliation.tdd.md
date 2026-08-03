# Broker order reconciliation — TDD evidence

## Safety defect

The live loop treated an accepted Alpaca request as a completed trade. A full
sell cleared position risk state immediately, and a restart forgot accepted
orders. That could produce duplicate or conflicting orders while the original
order was still open or partially filled.

The bot now persists submitted order IDs and reconciles them with Alpaca at the
start of every market cycle. Any unresolved order blocks both buys and sells for
its symbol. Full-exit state and stop cooldowns change only after a confirmed
fill; canceled, expired, or rejected orders are resolved without pretending the
position closed.

## RED

The focused test collection failed before the implementation because the broker
adapter had no normalized order-status contract:

```text
ImportError: cannot import name 'BrokerOrderStatus' from
'src.broker.alpaca_client'
```

The RED checkpoint is commit `472d58e`.

## GREEN

Focused reconciliation suite:

```text
12 passed, 6 warnings in 1.00s
```

Full regression suite:

```text
83 passed, 7 warnings in 18.71s
```

Coverage across executable lines changed since the RED checkpoint is 88.0%
(110/125). The four broad production modules are 77% overall; that lower number
includes pre-existing broker, scheduling, and signal paths outside this change.
The warnings are dependency deprecations and the existing saved-XGBoost-model
compatibility notice.

## Guarantees

| Guarantee | Test |
| --- | --- |
| Submitted order IDs and metadata survive process restart | `test_pending_order_ledger_survives_restart` |
| Partial fills and broker lookup failures keep the symbol blocked | `test_partial_fill_remains_pending_and_blocks_conflicting_orders`, `test_status_lookup_failure_keeps_order_blocked` |
| Rejected full exits preserve high-water and cooldown state | `test_rejected_order_is_resolved_without_clearing_position_state` |
| A filled stop starts its cooldown; an ordinary filled exit does not | `test_filled_stop_exit_starts_cooldown_and_resolves_pending_order`, `test_filled_non_stop_exit_clears_highwater_without_cooldown` |
| Stops and portfolio-guard exits do not duplicate pending closes | `test_stop_scan_skips_pending_symbol_and_returns_newly_submitted_symbol`, `test_drawdown_guard_does_not_duplicate_close_for_pending_symbol` |
| Alpaca close and status responses retain the order ID and normalized status | `test_broker_close_returns_the_submitted_order_id`, `test_broker_order_status_normalizes_the_alpaca_response` |
| The live intent loop submits a trim once and blocks the next conflicting cycle | `test_live_trade_cycle_routes_cap_trim_as_partial_sell_order` |

## Operational behavior

Order submission is logged as `SUBMIT`. A later cycle logs `FILLED` or the
terminal broker status. If Alpaca status lookup is unavailable, the ledger entry
is retained and the symbol remains blocked, favoring missed trades over duplicate
orders.
