# Quote recovery TDD evidence

Date: 2026-07-30

## Behavior

- Keep the existing batched yfinance close fetch as the primary source.
- Send only missing or invalid symbols to Alpaca's multi-symbol latest-trade
  endpoint on the IEX feed.
- Accept only finite, positive prices from either source.
- Log and skip symbols still unresolved after fallback.
- Continue the trade cycle when either provider fails.

## RED

The first focused run failed before production changes:

```text
FAILED test_last_prices_recovers_invalid_quotes_with_fallback
TypeError: _last_prices() got an unexpected keyword argument 'fallback'

FAILED test_alpaca_latest_prices_uses_iex_and_filters_invalid_trades
AttributeError: 'AlpacaBroker' object has no attribute 'latest_prices'

2 failed
```

## GREEN

Focused quote-recovery and wiring tests:

```text
7 passed, 20 deselected
```

Full offline suite:

```text
33 passed
```

Python bytecode compilation and `git diff --check` also completed
successfully.

Changed-line coverage for production code:

```text
src/broker/alpaca_client.py: 21/22 added executable lines covered (95.5%)
src/trader.py:               23/24 added executable lines covered (95.8%)
```

The suite emits two unrelated known warnings: the serialized XGBoost model was
created with an older XGBoost version, and `websockets.legacy` is deprecated.
